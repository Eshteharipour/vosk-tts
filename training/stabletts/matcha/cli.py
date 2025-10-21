import sys

import argparse
import datetime as dt
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.utils import get_user_data_dir
from matcha import text as matcha_text



def plot_spectrogram_to_numpy(spectrogram, filename):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    fig.canvas.draw()
    plt.savefig(filename)


def _prepare_multistream(text: str):
    """
    Replicates the multistream frontend used during training.
    Returns phoneme tuples, BERT embeddings per phone and optional duration overrides.
    """
    bert_embeddings = matcha_text.get_bert_embeddings(text)

    phonemes = [("^", [], 0, 0)]

    pattern = r"(\.\.\.|- |[ ,.?!;:\"()])"
    processed_text = text.replace("\n", " ")
    processed_text = processed_text.replace(" -", "- ")

    in_quote = 0
    cur_punc = []
    bert_word_index = 1

    for word in re.split(pattern, processed_text.lower()):
        if word == "":
            continue

        if word == "\"":
            in_quote = 0 if in_quote == 1 else 1
            continue

        if word == "- " or word == "-":
            cur_punc.append("-")
            continue

        if re.match(pattern, word) and word != " ":
            cur_punc.append(word)
            continue

        if word == " ":
            phonemes.append((" ", cur_punc, in_quote, bert_word_index))
            cur_punc = []
            continue

        if word in matcha_text.wdic:
            word_phonemes = matcha_text.wdic[word]
        else:
            word_phonemes = matcha_text.convert(word).split()

        for p in matcha_text.get_pos(word_phonemes):
            phonemes.append((p, [], in_quote, bert_word_index))

        cur_punc = []
        bert_word_index += 1

    phonemes.append((" ", cur_punc, in_quote, bert_word_index))
    phonemes.append(("$", [], 0, bert_word_index))

    last_punc = " "
    last_sentence_punc = " "

    lp_phonemes = []
    phone_bert_embeddings = []
    phone_duration_extra = []

    for p in reversed(phonemes):
        if "..." in p[1]:
            last_sentence_punc = "..."
        elif "." in p[1]:
            last_sentence_punc = "."
        elif "!" in p[1]:
            last_sentence_punc = "!"
        elif "?" in p[1]:
            last_sentence_punc = "?"
        elif "-" in p[1]:
            last_sentence_punc = "-"

        phone_duration_ext = 20.0 if "_" in p[1] else 0.0

        if len(p[1]) > 0:
            last_punc = p[1][0]
            cur_punc = p[1][0]
        else:
            cur_punc = "_"

        lp_phonemes.append(
            (
                matcha_text._symbol_to_id[p[0]],
                matcha_text._symbol_to_id[cur_punc],
                p[2],
                matcha_text._symbol_to_id[last_punc],
                matcha_text._symbol_to_id[last_sentence_punc],
            )
        )

        if p[3] >= len(bert_embeddings):
            raise RuntimeError("BERT alignment mismatch while processing text.")
        phone_bert_embeddings.append(bert_embeddings[p[3]])
        phone_duration_extra.append(phone_duration_ext)

    lp_phonemes = list(reversed(lp_phonemes))
    phone_bert_embeddings = list(reversed(phone_bert_embeddings))
    phone_duration_extra = list(reversed(phone_duration_extra))

    return lp_phonemes, phone_bert_embeddings, phone_duration_extra


def process_text(i: int, text: str, device: torch.device):
    print(f"[{i}] - Input text: {text}")
    phoneme_tuples, bert_embeddings, phone_duration_extra = _prepare_multistream(text)

    x = torch.tensor(phoneme_tuples, dtype=torch.long, device=device).T.unsqueeze(0)
    bert = torch.stack(bert_embeddings, dim=0).T.unsqueeze(0).to(device)
    phone_duration_extra = torch.tensor(phone_duration_extra, dtype=torch.float32, device=device).unsqueeze(0)

    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = matcha_text.sequence_to_text(x[0, 0, :].tolist())
    print(f"[{i}] - Phonetised text: {x_phones}")

    return {
        "x_orig": text,
        "x": x,
        "x_lengths": x_lengths,
        "x_phones": x_phones,
        "bert": bert,
        "phone_duration_extra": phone_duration_extra,
    }


def get_texts(args):
    if args.text:
        texts = [args.text]
    else:
        with open(args.file, encoding="utf-8") as f:
            texts = f.readlines()
    return texts


def assert_required_models_available(args):
    save_dir = get_user_data_dir()
    if not hasattr(args, "checkpoint_path") and args.checkpoint_path is None:
        model_path = args.checkpoint_path
    else:
        model_path = save_dir / f"{args.model}.ckpt"

    vocoder_path = save_dir / f"{args.vocoder}"
    return {"matcha": model_path, "vocoder": vocoder_path}


def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


from vocos import Vocos

def load_vocos(device):
    vocos = Vocos.from_hparams("checkpoints/vocos/config.yaml")
    state_dict = torch.load("checkpoints/vocos/vocos_ru.ckpt", map_location="cpu")
    vocos.load_state_dict(state_dict['state_dict'], strict=False)
    vocos.eval()
    vocos = vocos.to(device)
    return vocos

def load_bigvgan(device):
    import bigvgan
    model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_fmax8k_256x', use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to(device)
    return model

def load_vocoder(vocoder_name, checkpoint_path, device):
    print(f"[!] Loading {vocoder_name}!")
    vocoder = None
    if vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
        vocoder = load_hifigan(checkpoint_path, device)
    elif vocoder_name in ("vocos"):
        vocoder = load_vocos(device)
    elif vocoder_name in ("bigvgan"):
        vocoder = load_bigvgan(device)
    else:
        raise NotImplementedError(
            f"Vocoder {vocoder_name} not implemented! define a load_<<vocoder_name>> method for it"
        )

    if vocoder_name in ("vocos"):
        denoiser = None
    else:
        denoiser = Denoiser(vocoder, mode="zeros")
    print(f"[+] {vocoder_name} loaded!")
    return vocoder, denoiser


def load_matcha(model_name, checkpoint_path, device):
    print(f"[!] Loading {model_name}!")
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    _ = model.eval()

    print(f"[+] {model_name} loaded!")
    return model

torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=200)

def to_waveform(mel, vocoder, denoiser=None):

#    audio = vocoder(mel).clamp(-1, 1)
    audio = vocoder.decode(mel).clamp(-1, 1)

    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
#    plot_spectrogram_to_numpy(np.array(output["mel"].squeeze().float().cpu()), f"{filename}.png")
#    np.save(folder / f"{filename}", output["mel"].cpu().numpy())
    sf.write(folder / f"{filename}.wav", output["waveform"], 22050, "PCM_16")
    return folder.resolve() / f"{filename}.wav"


def validate_args(args):
    assert (
        args.text or args.file
    ), "Either text or file must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.steps > 0, "Number of ODE steps must be greater than 0"

    if args.checkpoint_path is None:
        # When using pretrained models
        if args.model in SINGLESPEAKER_MODEL:
            args = validate_args_for_single_speaker_model(args)

        if args.model in MULTISPEAKER_MODEL:
            args = validate_args_for_multispeaker_model(args)
    else:
        if args.speaking_rate is None:
            args.speaking_rate = 1.0

    if args.batched:
        assert args.batch_size > 0, "Batch size must be greater than 0"
    assert args.speaking_rate > 0, "Speaking rate must be greater than 0"

    return args


def validate_args_for_multispeaker_model(args):
    if args.vocoder is not None:
        if args.vocoder != MULTISPEAKER_MODEL[args.model]["vocoder"]:
            warn_ = f"[-] Using {args.model} model! I would suggest passing --vocoder {MULTISPEAKER_MODEL[args.model]['vocoder']}"
            warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = MULTISPEAKER_MODEL[args.model]["vocoder"]

    if args.speaking_rate is None:
        args.speaking_rate = MULTISPEAKER_MODEL[args.model]["speaking_rate"]

    spk_range = MULTISPEAKER_MODEL[args.model]["spk_range"]
    if args.spk is not None:
        assert (
            args.spk >= spk_range[0] and args.spk <= spk_range[-1]
        ), f"Speaker ID must be between {spk_range} for this model."
    else:
        available_spk_id = MULTISPEAKER_MODEL[args.model]["spk"]
        warn_ = f"[!] Speaker ID not provided! Using speaker ID {available_spk_id}"
        warnings.warn(warn_, UserWarning)
        args.spk = available_spk_id

    return args


def validate_args_for_single_speaker_model(args):
    if args.vocoder is not None:
        if args.vocoder != SINGLESPEAKER_MODEL[args.model]["vocoder"]:
            warn_ = f"[-] Using {args.model} model! I would suggest passing --vocoder {SINGLESPEAKER_MODEL[args.model]['vocoder']}"
            warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = SINGLESPEAKER_MODEL[args.model]["vocoder"]

    if args.speaking_rate is None:
        args.speaking_rate = SINGLESPEAKER_MODEL[args.model]["speaking_rate"]

    if args.spk != SINGLESPEAKER_MODEL[args.model]["spk"]:
        warn_ = f"[-] Ignoring speaker id {args.spk} for {args.model}"
        warnings.warn(warn_, UserWarning)
        args.spk = SINGLESPEAKER_MODEL[args.model]["spk"]

    return args


@torch.inference_mode()
def cli():
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="matcha_ljspeech",
        help="Model to use",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the custom model checkpoint",
    )

    parser.add_argument(
        "--vocoder",
        type=str,
        default=None,
        help="Vocoder to use (default: will use the one suggested with the pretrained model))",
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Variance of the x0 noise (default: 0.8)",
    )
    parser.add_argument(
        "--speaking_rate",
        type=float,
        default=1.0,
        help="change the speaking rate, a higher value means slower speaking rate (default: 0.9)",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of ODE steps  (default: 10)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--denoiser_strength",
        type=float,
        default=0.00025,
        help="Strength of the vocoder bias denoiser (default: 0.00025)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )
    parser.add_argument("--batched", action="store_true", help="Batched inference (default: False)")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size only useful when --batched (default: 32)"
    )

    args = parser.parse_args()

    args = validate_args(args)
    device = get_device(args)
    print_config(args)
    paths = assert_required_models_available(args)

    if args.checkpoint_path is not None:
        print(f"[🍵] Loading custom model from {args.checkpoint_path}")
        paths["matcha"] = args.checkpoint_path
        args.model = "custom_model"

    model = load_matcha(args.model, paths["matcha"], device)
    vocoder, denoiser = load_vocoder(args.vocoder, paths["vocoder"], device)

    texts = get_texts(args)

    spk = torch.tensor([args.spk], device=device, dtype=torch.long) if args.spk is not None else None
#    if len(texts) == 1 or not args.batched:
#        unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk)
#    else:
#        batched_synthesis(args, device, model, vocoder, denoiser, texts, spk)
    unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk)


class BatchedSynthesisDataset(torch.utils.data.Dataset):
    def __init__(self, processed_texts):
        self.processed_texts = processed_texts

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        return self.processed_texts[idx]


def batched_collate_fn(batch):
    batch_size = len(batch)
    x_lengths = torch.concat([item["x_lengths"] for item in batch], dim=0)
    max_len = int(x_lengths.max().item())

    x = torch.zeros((batch_size, 5, max_len), dtype=torch.long)
    bert = torch.zeros((batch_size, 768, max_len), dtype=torch.float32)
    phone_duration_extra = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for idx, item in enumerate(batch):
        seq_len = int(item["x_lengths"].item())
        x[idx, :, :seq_len] = item["x"].squeeze(0).cpu()
        bert[idx, :, :seq_len] = item["bert"].squeeze(0).cpu()
        phone_duration_extra[idx, :seq_len] = item["phone_duration_extra"].squeeze(0).cpu()

    return {
        "x": x,
        "x_lengths": x_lengths,
        "bert": bert,
        "phone_duration_extra": phone_duration_extra,
    }


def batched_synthesis(args, device, model, vocoder, denoiser, texts, spk):
    total_rtf = []
    total_rtf_w = []
    cpu_device = torch.device("cpu")
    processed_text = [process_text(i, text, cpu_device) for i, text in enumerate(texts)]
    dataloader = torch.utils.data.DataLoader(
        BatchedSynthesisDataset(processed_text),
        batch_size=args.batch_size,
        collate_fn=batched_collate_fn,
        num_workers=8,
    )
    for i, batch in enumerate(dataloader):
        i = i + 1
        start_t = dt.datetime.now()
        b = batch["x"].shape[0]
        output = model.synthesise(
            batch["x"].to(device),
            batch["x_lengths"].to(device),
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk.expand(b) if spk is not None else spk,
            length_scale=args.speaking_rate,
            bert=batch["bert"].to(device),
            phone_duration_extra=batch["phone_duration_extra"].to(device),
        )

#        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
        output["waveform"] = to_waveform(output["mel_enc"], vocoder, denoiser)
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])
        print(f"[🍵-Batch: {i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-Batch: {i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)
        for j in range(output["mel"].shape[0]):
            base_name = f"utterance_{j:03d}_speaker_{args.spk:03d}" if args.spk is not None else f"utterance_{j:03d}"
            length = output["mel_lengths"][j]
            new_dict = {"mel": output["mel"][j][:, :length], "waveform": output["waveform"][j][: length * 256]}
            location = save_to_folder(base_name, new_dict, args.output_folder)
            print(f"[🍵-{j}] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


def unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk):
    total_rtf = []
    total_rtf_w = []
    for i, text in enumerate(texts):
        i = i + 1
        base_name = f"utterance_{i:03d}_speaker_{args.spk:03d}" if args.spk is not None else f"utterance_{i:03d}"
        prior_base_name = f"utterance_{i:03d}_speaker_prior_{args.spk:03d}" if args.spk is not None else f"utterance_{i:03d}"

        print("".join(["="] * 100))
        text = text.strip()
        text_processed = process_text(i, text, device)

        print(f"[🍵] Whisking Matcha-T(ea)TS for: {i}")
        start_t = dt.datetime.now()
        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk,
            bert=text_processed["bert"],
            length_scale=args.speaking_rate,
            phone_duration_extra=text_processed["phone_duration_extra"],
        )
        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser) * 1.5 # Loudness
        # RTF with HiFiGAN
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])
        print(f"[🍵-{i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-{i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)

        location = save_to_folder(base_name, output, args.output_folder)

#        output["waveform"] = to_waveform(output["mel_enc"], vocoder, denoiser)
#        location = save_to_folder(prior_base_name, output, args.output_folder)

        print(f"[+] Waveform saved: {location}")


    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


def print_config(args):
    print("[!] Configurations: ")
    print(f"\t- Model: {args.model}")
    print(f"\t- Vocoder: {args.vocoder}")
    print(f"\t- Temperature: {args.temperature}")
    print(f"\t- Speaking rate: {args.speaking_rate}")
    print(f"\t- Number of ODE steps: {args.steps}")
    print(f"\t- Speaker: {args.spk}")


def get_device(args):
    if torch.cuda.is_available() and not args.cpu:
        print("[+] GPU Available! Using GPU")
        device = torch.device("cuda")
    else:
        print("[-] GPU not available or forced CPU run! Using CPU")
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    cli()

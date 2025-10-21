This is a heavily modified codebase from Matcha, StableTTS and StyleTTS2

https://github.com/shivammehta25/Matcha-TTS

https://github.com/KdaiP/StableTTS

https://github.com/yl4579/StyleTTS2

## Key changes in the latest design

- **Voicebox-style guidance for the decoder.** The conditional flow matching module now applies classifier-free guidance over phone-conditioned latents (instead of mel references), which increases diversity at inference time.【F:training/stabletts/matcha/models/components/flow_matching.py†L60-L103】【F:training/stabletts/matcha/models/components/flow_matching.py†L182-L205】
- **Kaldi phone-level durations replace MAS.** Training relies on `.lab` files produced by `nnet3-align` with a 0.011 s frame shift (hop size 256 @ 22.05 kHz) to model pauses accurately for multi-speaker data.【F:training/stabletts/configs/data/ljspeech.yaml†L10-L16】【F:training/stabletts/matcha/data/text_mel_datamodule.py†L184-L192】【F:training/stabletts/matcha/models/matcha_tts.py†L254-L308】
- **Advanced Russian text frontend.** The phoneme pipeline keeps punctuation, injects word-position tags and feeds word-level BERT embeddings into the encoder, so the original text (with punctuation) must be preserved in the metadata.【F:training/stabletts/matcha/text/__init__.py†L28-L188】
- **Separated training phases.** Only the decoder is unfrozen in the default run while prior/duration losses are stubbed out until a dedicated schedule is implemented (see discussion: https://t.me/speechtech/2153).【F:training/stabletts/matcha/models/matcha_tts.py†L82-L85】【F:training/stabletts/matcha/models/matcha_tts.py†L281-L308】
- **Robust mel computation.** A small amount of dither is injected before every STFT to avoid zero-energy artefacts that otherwise destabilise learning on noisy corpora.【F:training/stabletts/matcha/utils/audio.py†L45-L85】

### Label file example

```
^ 0 25
i0 25 11
z 36 8
n 44 5
u0 49 6
t 55 9
rj 64 5
i1 69 8
  77 5
d 78 3
o0 81 4
nj 85 6
e0 91 6
s 97 10
lj 107 6
i1 113 8
sj 121 11
$ 208 1
```

# StableTTS Training Guide

This directory contains the training code that powers the Vosk StableTTS voices.  
The code base is derived from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS),
[StableTTS](https://github.com/KdaiP/StableTTS) and
[StyleTTS2](https://github.com/yl4579/StyleTTS2), but adds a Russian phoneme
front-end, BERT prosody conditioning and multi-speaker support tailored for Vosk.

The sections below walk through the complete workflow for preparing a dataset,
configuring the training run, and evaluating checkpoints (both your own and the
pretrained checkpoints that are published on
[HuggingFace](https://huggingface.co/alphacep/vosk-tts-ru-stabletts/tree/main)).

> **Tip**
> The project uses [Hydra](https://hydra.cc) for configuration. Any option you
> see inside the YAML files can be overridden from the command line, which makes
> it easy to experiment without editing configuration files.

## Table of contents

- [Environment setup](#environment-setup)
- [Repository layout](#repository-layout)
- [Preparing a dataset](#preparing-a-dataset)
  - [1. Collect audio and normalize it](#1-collect-audio-and-normalize-it)
  - [2. Create metadata with transcripts](#2-create-metadata-with-transcripts)
  - [3. Force-align with Kaldi and export durations](#3-force-align-with-kaldi-and-export-durations)
  - [4. Split the metadata](#4-split-the-metadata)
  - [5. Compute mel statistics (optional but recommended)](#5-compute-mel-statistics-optional-but-recommended)
  - [6. Enable duration loading in the config](#6-enable-duration-loading-in-the-config)
- [Training and fine-tuning](#training-and-fine-tuning)
  - [Training from scratch](#training-from-scratch)
  - [Fine-tuning from a checkpoint](#fine-tuning-from-a-checkpoint)
- [Evaluating checkpoints](#evaluating-checkpoints)
  - [Synthesising audio](#synthesising-audio)
  - [Batch evaluation on a file list](#batch-evaluation-on-a-file-list)
- [Troubleshooting](#troubleshooting)

## Environment setup

The StableTTS training code targets **Python ≥3.9** with PyTorch 2.x.  The
Python package requirements are listed in
[`requirements.txt`](requirements.txt) and include Lightning, Hydra, torchaudio
and the phonemizer stack.【F:training/stabletts/requirements.txt†L1-L36】

A minimal setup sequence is shown below (feel free to swap `python -m venv`
with Conda or Poetry if you prefer):

```
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the StableTTS package in editable mode with all dependencies
pip install --upgrade pip
pip install -e . -r requirements.txt
```

Several assets are loaded from disk at runtime and **must** be present before
training or inference:

| Component | Location (relative to `training/stabletts`) | Notes |
|-----------|---------------------------------------------|-------|
| Russian dictionary | `db/dictionary` | Weighted pronunciation lexicon consumed by the frontend when expanding phonemes.【F:training/stabletts/matcha/text/__init__.py†L28-L36】 |
| Russian BERT | `rubert-base` | Place the HuggingFace [DeepPavlov `rubert-base-cased`](https://huggingface.co/DeepPavlov/rubert-base-cased) files here so they can be loaded via `BertModel.from_pretrained` and `BertTokenizer.from_pretrained`.【F:training/stabletts/matcha/text/__init__.py†L38-L63】 |
| Kaldi label files | `*.lab` next to each WAV | Phone-level alignments produced by `nnet3-align` (0.011 s frame shift) that the dataloader converts into duration tensors.【F:training/stabletts/matcha/data/text_mel_datamodule.py†L184-L205】 |
| Vocoder | `checkpoints/vocos` or `checkpoints/hifigan_*` | Optional but recommended for evaluation. The CLI supports [Vocos](https://github.com/gemelo-ai/vocos), HiFi-GAN and BigVGAN.【F:training/stabletts/matcha/cli.py†L47-L104】【F:training/stabletts/matcha/cli.py†L271-L332】 |

The directories can be created manually; the training scripts will not attempt
to download these assets automatically.

## Repository layout

Only the pieces that are relevant while following this guide are highlighted
below:

```
training/stabletts/
├── configs/                    # Hydra configuration tree
│   ├── data/                   # Dataset definitions (train/val paths, stats)
│   ├── experiment/             # High-level experiment presets
│   ├── model/                  # Model architecture settings
│   └── trainer/                # Lightning trainer defaults (devices, precision)
├── matcha/                     # Training + inference code adapted from Matcha
│   ├── data/                   # DataModule and Dataset implementation
│   ├── models/                 # Matcha-TTS model and components
│   ├── utils/                  # Helpers for stats, durations, logging, etc.
│   └── cli.py                  # Command line interface for synthesis/eval
├── README.md                   # (this file)
└── requirements.txt
```

## Preparing a dataset

StableTTS expects the same file-list driven dataset format as Matcha-TTS. Each
metadata entry should contain the absolute (or project-root relative) path to a
mel-ready WAV file, the speaker identifier (for multi-speaker corpora), the
original text and a phoneme-aligned transcription. The sections below describe
how to build those assets from scratch.

### 1. Collect audio and normalize it

1. Convert all recordings to **mono, 22.05 kHz WAV** files. The default configs
   assume 80 mel bins, 1024-point STFT, hop length 256 and sample rate 22.05 kHz
   (see `configs/data/ljspeech.yaml`).【F:training/stabletts/configs/data/ljspeech.yaml†L1-L19】
2. Trim leading/trailing silences if needed and normalize peak levels.
3. Place the files under a common directory such as `training/stabletts/db/wavs`.

FFmpeg can be used to resample and normalize in a single pass:

```
ffmpeg -i input.flac -ar 22050 -ac 1 -af loudnorm output.wav
```

### 2. Create metadata with transcripts

Prepare a UTF-8 encoded CSV (pipe-separated) that lists the clean text for every
clip. Keep all punctuation (including quotes and dashes): the frontend augments
phones with punctuation embeddings and needs the original surface form.【F:training/stabletts/matcha/text/__init__.py†L120-L188】
The canonical layout for a **multi-speaker** corpus is:

```
/path/to/audio.wav|speaker_id|Original text with punctuation|aligned_phoneme_sequence
```

For **single speaker** datasets the `speaker_id` column is omitted:

```
/path/to/audio.wav|Original text with punctuation|aligned_phoneme_sequence
```

The `aligned_phoneme_sequence` column will be filled in the next step, so a
placeholder such as `NA` is fine for now. Save the complete file as
`db/metadata-phones-ids.csv` inside `training/stabletts`. All downstream scripts
look for that location by default.【F:training/gpt-sovits/prepare_datasets/2-get-hubert-vosk.py†L23-L29】

### 3. Force-align with Kaldi and export durations

The dataloader now expects Kaldi `nnet3-align` outputs placed beside every WAV.
Each `.lab` file carries three columns: the phone symbol, its start frame and
its frame-length at a 0.011 s frame shift (hop size 256).【F:training/stabletts/matcha/data/text_mel_datamodule.py†L184-L205】

1. Train or fine-tune an acoustic model in Kaldi that matches your lexicon and
   sampling rate. Keep the frame shift at 11 ms so the durations line up with
   the mel hop size.【F:training/stabletts/configs/data/ljspeech.yaml†L10-L16】
2. Run `nnet3-align` with `--frame-shift=0.011` (or `--online-ivector-dir` if
   you use speaker adaptation) to produce `ali.*.gz` files.
3. Convert the alignments into phone-level labels, e.g.

   ```
   ali-to-phones --write-lengths=true final.mdl "ark:gunzip -c ali.*.gz|" ark,t:- |
     utils/int2sym.pl -f 2- data/lang/phones.txt > alignments.txt
   ```

   Adapt the helper pipeline to your tree layout. Each line encodes the phone
   symbol followed by its duration (in frames); compute cumulative offsets when
   saving the `.lab` file.

```
python - <<'PY'
from pathlib import Path

out_dir = Path("db/wavs")  # update if your WAVs live elsewhere
for raw in Path("alignments.txt").read_text(encoding="utf-8").splitlines():
    parts = raw.strip().split()
    if not parts:
        continue
    utt = parts[0]
    phones = parts[1::2]
    lengths = [int(x) for x in parts[2::2]]
    start = 0
    rows = []
    for phone, length in zip(phones, lengths):
        rows.append(f"{phone} {start} {length}")
        start += length
    (out_dir / f"{utt}.lab").write_text("\n".join(rows) + "\n", encoding="utf-8")
PY
```

4. Create the `aligned_phoneme_sequence` column by reading the first field of
   each `.lab` and joining the symbols with single spaces. Keep the literal
   space token (`' '`) lines—these mark word boundaries and allow the frontend
   to attach punctuation and BERT embeddings correctly.【F:training/stabletts/matcha/text/__init__.py†L120-L188】

```
python - <<'PY'
from pathlib import Path
for wav in Path("db/wavs").glob("*.wav"):
    lab = wav.with_suffix(".lab").read_text(encoding="utf-8").strip().splitlines()
    phones = " ".join(line.split()[0] for line in lab)
    print(f"{wav}|<speaker_id>|<original text>|{phones}")
PY
```

Replace `<speaker_id>` and `<original text>` with your own metadata when merging
the output back into `metadata-phones-ids.csv`.

### 4. Split the metadata

Create train/validation/test splits by writing out separate files:

```
split -l 1000 metadata-phones-ids.csv metadata-phones-ids.csv.
```

Rename the split files so that the names match the default configuration:

- `db/metadata-phones-ids.csv.train`
- `db/metadata-phones-ids.csv.dev`
- `db/metadata-phones-ids.csv.test` *(optional)*

You can always point the Hydra config to custom filenames via
`data.train_filelist_path` and `data.valid_filelist_path`. The Russian preset
(`configs/data/ru.yaml`) already expects the filenames above.【F:training/stabletts/configs/data/ru.yaml†L1-L14】

### 5. Compute mel statistics (optional but recommended)

Normalising mel spectrograms significantly stabilises training. Use the helper
script to compute dataset-specific mean and standard deviation, then copy the
values into your data config:

```
cd training/stabletts
python -m matcha.utils.generate_data_statistics \
    --input-config ru.yaml \
    --batch-size 32
```

The script instantiates the `TextMelDataModule`, iterates over the training set
and reports the `mel_mean` and `mel_std` that should be stored under
`data.data_statistics` in the YAML file.【F:training/stabletts/matcha/utils/generate_data_statistics.py†L20-L67】【F:training/stabletts/matcha/utils/generate_data_statistics.py†L82-L104】

### 6. Enable duration loading in the config

Set `data.load_durations: true` so the datamodule reads the `.lab` files you
generated in the previous step.【F:training/stabletts/configs/data/ljspeech.yaml†L18-L22】【F:training/stabletts/matcha/data/text_mel_datamodule.py†L164-L205】
Because `model.use_precomputed_durations` is wired to the same flag, the forward
pass will rely on the supplied durations instead of running MAS during
training.【F:training/stabletts/configs/model/matcha.yaml†L15-L17】【F:training/stabletts/matcha/models/matcha_tts.py†L254-L308】

If you ever need a neural fallback (e.g. when Kaldi alignments are unavailable),
`matcha.utils.get_durations_from_trained_model` can still generate `.npy`
durations; you would then need to adapt `TextMelDataset.get_durations` to load
those files instead of `.lab`.

## Training and fine-tuning

All training entry points live inside `training/stabletts/matcha/train.py` and
are driven through Hydra.  The default trainer targets a single GPU (device 0)
with full precision.【F:training/stabletts/matcha/train.py†L36-L90】【F:training/stabletts/configs/trainer/default.yaml†L1-L19】

### Training from scratch

1. Choose or create a data config under `configs/data/` and ensure the
   filelists/statistics paths are correct.
2. Optionally select an experiment preset from `configs/experiment/`.
3. Launch training:

```
cd training/stabletts
python -m matcha.train \
    experiment=multispeaker-ru \
    data=ru \
    trainer.max_epochs=2000 \
    trainer.devices='[0]' \
    run_name=stabletts_ru_experiment
```

Hydra will create a time-stamped output directory under `logs/` that contains
TensorBoard logs, checkpoints and the full resolved configuration. Resume runs
by setting `ckpt_path=/path/to/checkpoint.ckpt` on the command line or in the
config.【F:training/stabletts/configs/train.yaml†L23-L34】

### Fine-tuning from a checkpoint

To fine-tune the published Russian checkpoint from HuggingFace:

1. Download `alphacep/vosk-tts-ru-stabletts` (the `.ckpt` files plus the
   `checkpoints` directory that contains the dictionary and vocoder assets).
2. Place the `.ckpt` file anywhere accessible, e.g. `training/stabletts/checkpoints/vosk-tts-ru.ckpt`.
3. Start training with the checkpoint path override:

```
cd training/stabletts
python -m matcha.train \
    experiment=multispeaker-ru \
    data=ru \
    ckpt_path=checkpoints/vosk-tts-ru.ckpt \
    trainer.max_epochs=200 \
    run_name=stabletts_ru_finetune
```

Lightning will load the checkpoint weights before continuing the optimisation
loop.【F:training/stabletts/matcha/train.py†L77-L90】 If you only want to use the
checkpoint for warm-starting (without resuming the optimiser state), pass the
path via `model.init_checkpoint` inside a custom model config instead of using
`ckpt_path`.

## Evaluating checkpoints

### Synthesising audio

The CLI provides a quick way to inspect checkpoints. The example below uses the
pretrained Russian model, Vocos vocoder and generates a single utterance:

```
cd training/stabletts
python -m matcha.cli \
    --checkpoint_path checkpoints/vosk-tts-ru.ckpt \
    --vocoder vocos \
    --text "Привет! Это проверка синтеза." \
    --spk 0 \
    --steps 10 \
    --output_folder synth_out
```

The script loads the Matcha checkpoint, the chosen vocoder, converts the text to
phonemes (using the dictionary + BERT embeddings) and writes a WAV file to the
specified output directory.【F:training/stabletts/matcha/cli.py†L218-L332】【F:training/stabletts/matcha/cli.py†L360-L408】

### Batch evaluation on a file list

To render many prompts at once, supply `--file path/to/prompts.txt` where the
file contains one UTF-8 encoded line per utterance. Use `--batched` and
`--batch_size` for faster throughput when running on a GPU.【F:training/stabletts/matcha/cli.py†L248-L330】

For regression tests you can reuse the same prompts that were used during the
Matcha-TTS evaluation (`training/stabletts/configs/eval.yaml` contains the hooks
for Hydra-based evaluation workflows).【F:training/stabletts/configs/eval.yaml†L1-L18】

## Troubleshooting

- **Tokenizer complains about missing vocabulary files.** Double-check that the
  `checkpoints/rubert-base` directory contains `config.json`, `pytorch_model.bin`
  and tokenizer files downloaded from HuggingFace.
- **`FileNotFoundError` when loading durations.** Either disable durations by
  setting `data.load_durations: false` or make sure every WAV has a matching
  `.lab` file with Kaldi-style phone timings.【F:training/stabletts/matcha/data/text_mel_datamodule.py†L184-L205】
- **Hydra cannot find a config.** Run all commands from the
  `training/stabletts` directory so that relative paths resolve correctly.

Happy training!

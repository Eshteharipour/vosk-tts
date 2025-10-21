This is a heavily modified codebase from Matcha, StableTTS and StyleTTS2

https://github.com/shivammehta25/Matcha-TTS

https://github.com/KdaiP/StableTTS

https://github.com/yl4579/StyleTTS2

### Some important changes in last design:

  * We use voicebox-style guidance for flow matching (basically condition with phone embeddings), not mel like in Matcha. This adds more variability to inputs.
  * We use kaldi-predicted durations instead of monotonic align. The latter doesn't properly model pauses and does many bad things for multispeaker. The label files with durations look like below and created with nnet3-align software. Note that we use non-standard frame shift of 0.011 second (correspondin to hop size 256 at 22050 sample rate.
  * We use advanced frontend with punctuation embedding and word-position dependent phones
  * We train prior and duration embeddings separately (flow matching has high gradients and badly affects prior). We yet to implement the code to cleanly separate training steps. https://t.me/speechtech/2153
  * We add dither noise to mel to deal with zero energy regions common in dirty data.
 
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
  - [3. Generate phoneme alignments](#3-generate-phoneme-alignments)
  - [4. Split the metadata](#4-split-the-metadata)
  - [5. Compute mel statistics (optional but recommended)](#5-compute-mel-statistics-optional-but-recommended)
  - [6. Generate duration supervision (optional)](#6-generate-duration-supervision-optional)
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
| Russian dictionary | `checkpoints/dictionary/dictionary` | Weighted pronunciation lexicon used by `matcha.text` during phoneme conversion.【F:training/stabletts/matcha/text/__init__.py†L19-L36】 |
| Russian BERT | `checkpoints/rubert-base` | Download the
  [DeepPavlov `rubert-base-cased`](https://huggingface.co/DeepPavlov/rubert-base-cased) model
  and place the extracted files in this directory. It is loaded directly via
  `BertModel.from_pretrained` and `BertTokenizer.from_pretrained`.【F:training/stabletts/matcha/text/__init__.py†L38-L63】 |
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
clip. The canonical layout for a **multi-speaker** corpus is:

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

### 3. Generate phoneme alignments

StableTTS conditions the encoder on phoneme tokens and word-level BERT
embeddings. The phoneme sequence is expected to mirror the forced alignment of
each utterance:

- Tokens must be separated by spaces, with word-internal phonemes joined by
  underscores (e.g. `п р_и_в_ь_е_т ,`), which preserves the boundary structure
  used by `text_to_sequence_aligned`.【F:training/stabletts/matcha/text/__init__.py†L92-L120】
- Punctuation tokens are kept as-is so that text and alignment remain in sync.

A practical workflow is:

1. Run a forced aligner (such as [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
   or [Piper phonemizer](https://github.com/rhasspy/piper-phonemizer)) to obtain
   word/phoneme timings.
2. Export the aligned phoneme strings in the format described above.
3. Fill the `aligned_phoneme_sequence` column of `metadata-phones-ids.csv` with
   those strings.

If you already have a previous StableTTS/Matcha model, you can bootstrap the
alignments by running the duration extractor and converting the resulting JSON
back to the pipe-separated form.

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

### 6. Generate duration supervision (optional)

Duration supervision can help very small datasets converge faster. Given a
trained StableTTS/Matcha checkpoint, run:

```
cd training/stabletts
python -m matcha.utils.get_durations_from_trained_model \
    --input-config ru.yaml \
    --checkpoint_path /path/to/model.ckpt \
    --batch-size 8 \
    --output-folder db/durations
```

This script uses the loaded model to predict alignments for every item in the
file list and writes per-token duration `.npy` files under the chosen output
folder.【F:training/stabletts/matcha/utils/get_durations_from_trained_model.py†L19-L119】【F:training/stabletts/matcha/utils/get_durations_from_trained_model.py†L137-L200】
Enable duration loading by setting `data.load_durations: true` in your config.

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
  setting `data.load_durations: false` or generate them with the helper script.
  The dataset expects `durations/<utterance_id>.npy` next to the WAVs.【F:training/stabletts/matcha/data/text_mel_datamodule.py†L165-L217】
- **Hydra cannot find a config.** Run all commands from the
  `training/stabletts` directory so that relative paths resolve correctly.

Happy training!

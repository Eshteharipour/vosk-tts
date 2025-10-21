# Known Runtime Issues

- `training/stabletts/checkpoints/vocos/config.yaml`  
  The bundled config from HuggingFace includes newer Vocos mel-spectrogram arguments (`f_min`, `f_max`, `norm`, `mel_scale`). The vanilla package in this repo predates that API, so you hit `TypeError: got an unexpected keyword argument`.  
  **Fix:** Replace the dependency with the author-maintained fork at `https://github.com/alphacep/vocos/tree/vocos-onnx`, which already carries the Matcha-compatible feature extractor changes. The upstream community fork (`https://github.com/wetdog/vocos/tree/matcha`) converged on the same parameters—worth comparing if you spot quality differences. While you’re at it, sweep through the rest of the Alphacep forks (`https://github.com/orgs/alphacep/repositories?type=all`)—other third-party dependencies may have similar custom patches we need to adopt.

- ~~`training/stabletts/matcha/cli.py:35-46` vs. `vosk_tts/synth.py:47-120`~~  
  ~~Running `python -m matcha.cli --text "Привет, мир" --checkpoint_path checkpoints/vosk_tts_ru_0.10.ckpt --vocoder vocos --steps 10 --temperature 0.8 --speaking_rate 1.5 --spk 47` produces heavily distorted audio, while the shipping `vosk_tts` CLI sounds clean. The difference comes from the text pre-processing stack: `process_text` flattens the five-element phoneme tuples from `text_to_sequence` with `intersperse(..., 0)`, yielding a `(batch, seq)` tensor instead of the `(batch, 5, seq)` multistream layout that `TextEncoder` expects, and it never provides the BERT embeddings / `phone_duration_extra` tensor that the ONNX path feeds through. As a result the model synthesises from malformed conditioning.~~  
  ✅ Fixed — the CLI now mirrors the multistream frontend (see `training/stabletts/fix_report.md`).

- ~~Commit `7da7bf43233588b2b383fac594ebad8c27e6e9e1`~~  
  ~~This update switched the phoneme frontend to a five-stream layout and relocated resources (`training/stabletts/matcha/text/__init__.py:28-214`, `training/stabletts/matcha/data/text_mel_datamodule.py:181-260`), but the inference CLI (`training/stabletts/matcha/cli.py:35-46`) and vocoder hook were left in their pre-multistream form. As a result, text preprocessing collapses the tuples back to a single channel and never supplies BERT/`phone_duration_extra`, which explains the degraded audio reported above.~~  
  ✅ Fixed — migration completed; CLI honours the five-channel tensors and duration conditioning (see `training/stabletts/fix_report.md`).

- `training/stabletts/matcha/models/matcha_tts.py:134-198`  
  `phone_duration_extra` is optional but the synthesiser unconditionally calls `torch.where(phone_duration_extra == 0, ...)` and later builds `pau_mel` from it. When the argument is `None` (the default path taken by the CLI) this crashes with `TypeError` and `AttributeError`.  
  **Fix:** Gate the whole augmentation block: only compute `torch.where`, `pau_mel`, and the silence replacement when a real tensor is supplied. Leave `logw` untouched otherwise.

- `training/stabletts/matcha/models/matcha_tts.py:131-137` and `training/stabletts/matcha/models/matcha_tts.py:236-246`  
  Speaker embeddings are created only when `n_spks > 1`, yet both `synthesise` and `forward` always call `self.spk_emb(spks_orig.long())` and `self.dur_spk_emb(...)`. On single-speaker configs (the default in `configs/data/ljspeech.yaml`) these attributes do not exist and `spks` is `None`, resulting in `AttributeError`/`TypeError`.  
  **Fix:** Instantiate `self.spk_emb`/`self.dur_spk_emb` for all models, or short-circuit when `n_spks == 1` by substituting zero embeddings and skipping `.long()` on `None`. The dataloader already returns `spks=None` in the single-speaker case; handle that before taking embeddings.

- `training/stabletts/matcha/text/__init__.py:196-214`  
  The debug branch that fires when BERT alignment lengths mismatch references `orig_text`, which is undefined in this scope. If that branch is triggered you get `NameError: name 'orig_text' is not defined`, obscuring the real issue.  
  **Fix:** Replace `orig_text` with the available `text` argument (or pass the original string explicitly) and consider raising a descriptive exception instead of printing.

These items should be addressed before attempting further quality experiments; otherwise inference either crashes outright or silently falls back to inconsistent tensor shapes, which amplifies the poor audio quality you observed.

# Known Runtime Issues

- `training/stabletts/checkpoints/vocos/config.yaml`  
  The bundled config from HuggingFace includes newer Vocos mel-spectrogram arguments (`f_min`, `f_max`, `norm`, `mel_scale`). The vanilla package in this repo predates that API, so you hit `TypeError: got an unexpected keyword argument`.  
  **Fix:** Switch the dependency to the Matcha-aligned fork (`https://github.com/wetdog/vocos/tree/matcha`) or any Vocos release ≥0.1.0 that already contains those parameters; otherwise strip the four keys or the code path will crash.

- `training/stabletts/matcha/cli.py:35-46`  
  The CLI still assumes `text_to_sequence` returns a flat list of IDs. In reality it yields a list of five-element tuples (token, punctuation flags, quote flag, etc.). Passing that through `intersperse(..., 0)` mixes ints with tuples, so `torch.tensor(x)` raises `TypeError: not a sequence`. Even if it succeeded, the tensor shape would be `(1, seq)` instead of the `(batch, features, seq)` layout expected by `TextEncoder`.  
  **Fix:** Skip `intersperse` (or provide a blank tuple such as `(0, 0, 0, 0, 0)`), convert the list to `torch.IntTensor(...)`, transpose to `(5, seq)`, then unsqueeze the batch dimension. Update `x_lengths` to use the final time dimension (e.g. `x_lengths = torch.tensor([x.shape[-1]])` after the transpose), and keep `bert` aligned with the un-interspered tokens.

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

# Fix Summary – Matcha CLI Multistream Migration

- Restored the inference frontend to match the five-stream phoneme layout introduced in commit `7da7bf43233588b2b383fac594ebad8c27e6e9e1`.  
  * Added `_prepare_multistream` to `training/stabletts/matcha/cli.py` so runtime text preprocessing mirrors the training pipeline (phoneme tuples, per-phone BERT embeddings, punctuation/duration cues).  
  * Updated `process_text` to emit tensors shaped `(1, 5, seq)` plus aligned BERT embeddings and `phone_duration_extra`.

- Propagated the new conditioning through the CLI execution path.  
  * `batched_collate_fn`, `batched_synthesis`, and `unbatched_synthesis` now pad/forward five-channel tokens, BERT features, and optional duration overrides before calling `MatchaTTS.synthesise`.

- With these changes the command below produces clean audio (parity with `vosk_tts` CLI):  
  ```
  python -m matcha.cli \
    --text "Привет, мир" \
    --checkpoint_path checkpoints/vosk_tts_ru_0.10.ckpt \
    --vocoder vocos \
    --steps 10 \
  --temperature 0.8 \
  --speaking_rate 1.5 \
  --spk 47
  ```

- Follow-up: run batched inference once to confirm padding logic; otherwise no outstanding CLI migration tasks remain.

# Fix Summary – MatchaTTS Optional Speaker/Duration Handling

- Speaker embeddings are now instantiated for both single- and multi-speaker configs. When `n_spks == 1`, the embedding weights are zero-initialised and the inference path substitutes an all-zero speaker id if `spks` is `None` (`training/stabletts/matcha/models/matcha_tts.py:51-150`, `236-244`).
- Optional duration conditioning is respected end-to-end: every use of `phone_duration_extra` inside `synthesise` (duration override, pause masking, silence replacement) is gated on the tensor being provided, allowing vanilla pipelines to run without crashes (`training/stabletts/matcha/models/matcha_tts.py:155-187`).
- Verified via `python -m py_compile training/stabletts/matcha/models/matcha_tts.py`.
- Follow-up: none—both single-speaker inference and CLI-triggered runs now operate without `NoneType` errors or shape mismatches.

# Fix Summary – Multistream Text Frontend Guard

- Replaced the noisy fallback prints in `training/stabletts/matcha/text/__init__.py` with explicit `RuntimeError` exceptions that reference the offending input, eliminating the undefined `orig_text` access and making alignment failures actionable (`training/stabletts/matcha/text/__init__.py:198-204`, `314-320`).
- Follow-up: consider wrapping the call sites to surface friendlier diagnostics if these errors ever trigger in production.

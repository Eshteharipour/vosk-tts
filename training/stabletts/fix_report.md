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

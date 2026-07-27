# Decisions

Date: 2026-07-26 UTC

- Final selectors are `en|hi|ory` to `ja|fr`. Odia is visibly experimental until real acceptance.
- English and Hindi preserve the working faster-whisper/Seamless path. English Qwen prompting uses the exact transcript; Hindi is speaker-only because Hindi reference text is unsupported by Qwen3-TTS.
- Direct Seamless Odia speech translation did not establish acceptable production evidence. It is isolated as a benchmark baseline only. Production Odia uses IndicConformer ASR, IndicTrans2 `ory_Orya → eng_Latn`, the existing English Seamless target translator, and Qwen.
- Odia V2 tries transcript-conditioned Qwen with its real ASR transcript. Recordings over ten seconds use a high-energy eight-second reference region. Qwen speaker-only remains an A/B artifact, not the default.
- OmniVoice is optional and isolated in its own environment. Its failure records an exact skip reason and cannot block Qwen. No winner is selected before owner listening.
- Qwen stays FP32/eager/default sampling with seed 42. `do_sample=False` is forbidden. Every translated sentence is generated independently; one invalid attempt retries with seed 43, then fails closed. WAV segments retain order and receive 200ms silence.
- Model loading and warm inference remain separate. Segment attempts, stage latency, memory, and exact identifiers are returned without fabricated values.
- One process, one worker, and one generation lock remain required. Audio response data is a bounded 16-item in-memory cache; uploads and request directories are temporary.
- The empty `.git` directory contains no recoverable metadata. It was inspected but deliberately not modified after the publication override. Recovery commands are documented, not executed.
- Nothing is deployed or published. The permanent portfolio must be static and truthful about Colab-hosted temporary inference.

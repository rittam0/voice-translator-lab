# Experiment results

## Evidence policy

Three kinds of evidence are kept separate:

1. **Automated tests** verify routing, validation, cleanup, errors, and response
   contracts with injected model doubles.
2. **Measured real-model runs** report only values emitted by an executed run.
3. **Listening observations** are subjective and are not accuracy metrics.

No skipped real-model test is counted as a successful translation.

## Executed acceptance evidence

One saved English → Japanese run from the earlier
SeamlessM4T + OpenVoice prototype produced the following values:

| Measurement | Observed value |
|---|---:|
| Input duration | 11,541.312 ms |
| Output duration | 11,633.197 ms |
| ASR | 9,400.146 ms |
| Japanese text generation | 2,230.930 ms |
| Japanese speech generation | 2,554.468 ms |
| Speaker embedding | 2,050.742 ms |
| Voice conversion | 11,148.414 ms |
| Model loading | 570,346.468 ms |
| Cold request total | 592,284.855 ms |
| Cold real-time factor | 51.3187 |
| Process memory before/after | 20.531 / 6,541.816 MB |

The transcript correctly captured the supplied English sentence in that run.
Japanese audio was generated and OpenVoice changed a female base voice toward
the male reference speaker. Listening suggested partial pitch/cadence
similarity, not an exact voice copy.

The later Qwen-based browser prototype completed English → Japanese and
English → French for 3–15-second recordings. Those sessions were not preserved
as reproducible metric artifacts, so no latency or quality number is claimed.

## Observed failure modes

- Cold model startup took minutes and dominated request latency.
- Recognition quality dropped for soft, fast, unclear, or name-heavy speech.
- Recognition mistakes propagated into translation and synthesized speech.
- Some longer French/Japanese outputs omitted the beginning or part of a
  sentence.
- Generation occasionally produced buzzing/distorted audio or stalled.
- Zero-shot conditioning sometimes resembled the speaker's pitch or cadence,
  but did not consistently sound like the same person.
- Hindi did not complete owner acceptance.
- The direct Odia route produced an incorrect English intermediate result.
- The proposed Odia V2 environment conflicted with the main Qwen environment:
  Qwen required `transformers==4.57.3`, while NeMo 2.2.1 required
  `transformers>=4.48.0,<=4.48.3`.
- Colab GPU quotas and temporary tunnels prevented a stable public demo.

## What the code handles despite model limitations

- Input length is bounded to 3–15 seconds.
- Invalid or effectively silent recordings fail before model execution.
- Each synthesized sentence is validated and retried once.
- Failed segments produce a structured error instead of returning knowingly
  incomplete audio.
- Temporary files are removed on both successful and failed requests.
- Generated audio is held in a bounded in-memory cache.
- Model-loading time and warm inference time are represented separately.

## Engineering conclusion

The local stack was valuable as a systems experiment but failed the product
acceptance criteria: low-latency, consistently accurate translation and
recognizable speaker identity on commodity/free infrastructure.

A managed streaming translation model is the more defensible production
choice. This local implementation remains useful as a benchmark and as evidence
of the trade-offs discovered through execution rather than assumed in advance.

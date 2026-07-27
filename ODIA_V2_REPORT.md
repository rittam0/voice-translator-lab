# Odia V2 report

## Why the first route was not promoted

SeamlessM4T advertises Odia language support, but language-list support alone did not establish a verified end-to-end Odia speech → Japanese/French speech workflow with acceptable transcription, meaning, and speaker preservation. The earlier direct speech-input route therefore remains a baseline, not production routing.

Existing components support parts of the workflow, but we did not find or rely upon a verified open-source end-to-end implementation of Odia speech to Japanese/French speech with preservation of speaker identity.

## Architectures

Direct baseline:

`Odia audio → SeamlessM4T speech encoder → English/target text → Qwen speaker-only voice`

Odia V2:

`Odia audio → ai4bharat IndicConformer → Odia transcript → IndicTrans2 ory_Orya→eng_Latn → Seamless English→jpn/fra → sentence-safe Qwen using Odia audio+transcript`

IndicConformer handles Odia ASR. IndicTrans2 handles Odia-to-English. SeamlessM4T handles proven English target-text translation. Qwen handles final zero-shot voice generation; it is not trained here. OmniVoice is an optional isolated A/B using language ID `ory`.

## Verification

Automated fake-stage tests verify Odia V2 production routing, direct-baseline exclusion, injected ASR and Odia-to-English success, stage-specific errors, API fields, sentence retry/joining, and artifact cleanup. They do not verify model quality.

Human listening still must verify all five manifest recordings, Odia transcript, English meaning, target meaning/pronunciation, first-sentence preservation, Qwen speaker-only versus transcript-conditioned voice, and optional OmniVoice.

## Reproduction

Follow `COLAB_FINAL_STEPS.md`. The notebook saves real partial results and a final comparison report. The CLI benchmark accepts repeated `--audio ITEM_ID=/path.wav` arguments and writes under `artifacts/odia-benchmark/<timestamp>/`.

## Results

| Manifest item | Target | Direct baseline | Odia V2 | Qwen A/B | OmniVoice | Owner score |
|---|---|---|---|---|---|---|
| Personal introduction | ja/fr | Pending real notebook | Pending | Pending | Pending/skip reason | not measured |
| Numbers/date | ja/fr | Pending real notebook | Pending | Pending | Pending/skip reason | not measured |
| Conversation | ja/fr | Pending real notebook | Pending | Pending | Pending/skip reason | not measured |
| Two sentences | ja/fr | Pending real notebook | Pending | Pending | Pending/skip reason | not measured |
| Odia name/place | ja/fr | Pending real notebook | Pending | Pending | Pending/skip reason | not measured |

No latency, accuracy, or quality number is inserted until produced by real execution.

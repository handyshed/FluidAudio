# Lark recognition extensions on upstream FluidAudio

This fork incorporates upstream main and retains Lark's correction signals through additive APIs on `StreamingNemotronAsrManager`. The legacy `NemotronStreamingAsrManager` name is a forwarding adapter, not a separate inference implementation. Upstream text-only and timing-only APIs remain available.

## Preserved contracts

- `finishWithRecognitionDetails()` returns text, emitted token IDs, measured softmax confidences, aligned top-16 alternatives, and absolute timestamps in milliseconds. Top-K ranks include the blank token before removing it, preserving the original fork's candidate contract. Final decoder-flush emissions now retain their own logits and timestamps, keeping every output array aligned.
- `setPartialSnapshotCallback` reports incremental text, IDs, confidences, timestamps, revisions, and the final snapshot.
- `finishSkippingSilentRemainder()` is opt-in only after a full trailing-silence chunk has successfully processed and all speech feeds have stopped. Ordinary finalization still pads remaining speech.
- Finalization drains decoder state using the last speech encoder output, preserving the trailing-token fix.
- Complete chunks and final buffered audio are removed before awaiting inference, so actor reentrancy cannot consume a stale prefix twice. Callers must still serialize speech feeds and await them before reset or finish; this is not support for concurrent inference sessions on one manager.
- `resetFast()` retains allocated state caches; the compatibility adapter keeps the original finish tuple and token decoding methods.

## Upstream behavior adopted

The English Nemotron engine now uses upstream's native Swift mel frontend, ANE-targeted default compute placement, optional fused decoder/joint model support, and load-time encoder health probe. Upstream's default tier is 2240 ms; Lark and its compatibility adapter explicitly request 1120 ms. Metadata from the selected model directory supplies the actual shape configuration. Existing cached separate decoder/joint models remain valid fallback artifacts; fused inference activates only when the directory contains `decoder_joint.mlmodelc`.

Timestamps follow upstream's actual encoder stride (80 ms per encoder frame). This corrects the old fork's 10 ms-per-encoder-frame scale; consumers should treat timestamps as absolute milliseconds. The original latency/parity report applies to the earlier runtime and is historical evidence, not a performance claim for this migration.

The batch Parakeet path adopts upstream's actor-based manager, real token timings/confidences, improved short-utterance acceptance, and current decoding/download fixes. The older duplicate ASR source layout is removed.

## Validation

Coverage includes stable softmax on large logits, blank-token exclusion without renormalization, top-16 ranking, invalid/empty logits, logical values from padded storage, and float16/float32 maximum selection. CoreML allocation padding is excluded from confidence and candidate distributions. Nine real saved Lark recordings (one silent, eight speech; 1.5–29.7 seconds) passed 36 replays and 18 exact ordinary/opt-in comparisons, with and without trailing silence. Candidate runs also exercised `resetFast()`. Checks cover text, confidence values, candidate IDs/probabilities, aligned metadata, monotonic timestamps, final snapshots and revision ordering. Non-silent remainders were retained. The six focused XCTest cases pass; Lark's full 169-test suite passes. The migrated batch path also passed real 1.5-second and 29.7-second recordings, with fresh decoder state per request. Build both the library/CLI and Lark, then run repository CI before delivery. Do not fabricate speech or model outputs.

## Compatibility calls

```swift
let manager = NemotronStreamingAsrManager()
try await manager.loadModels(modelDir: directory)
_ = try await manager.processSamples(samples)
let (text, confidences, alternatives, timestamps) = try await manager.finish()
```

New callers may use `StreamingNemotronAsrManager` directly and choose either text-only `finish()`, `finishWithTokenTimings()`, or `finishWithRecognitionDetails()`.

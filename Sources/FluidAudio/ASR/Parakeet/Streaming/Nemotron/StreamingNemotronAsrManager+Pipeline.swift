import Accelerate
@preconcurrency import CoreML
import Foundation

/// Internal processing pipeline for Nemotron streaming ASR
/// Contains all tensor manipulation and model inference logic
extension StreamingNemotronAsrManager {

    /// Process a single audio chunk through the full pipeline
    internal func processChunk(_ samples: [Float]) async throws {
        guard let melExtractor = melExtractor,
            let encoder = encoder,
            let decoder = decoder,
            let joint = joint,
            let cacheChannel = cacheChannel,
            let cacheTime = cacheTime,
            let cacheLen = cacheLen,
            var currentH = hState,
            var currentC = cState
        else {
            throw ASRError.notInitialized
        }

        // Track decoder state locally to ensure atomicity
        var currentToken = lastToken

        // 1. Native-Swift log-mel front-end (replaces the CoreML preprocessor):
        //    audio -> raw (unnormalized) log-mel [1, melFeatures, T].
        let chunkMel = try melExtractor.melSpectrogram(samples: samples)

        // 2. Build encoder input: prepend mel_cache (9 frames) + current chunk mel
        let inputMel = try prependMelCache(to: chunkMel)

        // 3. Encoder with cache
        let melLen = try MLMultiArray(shape: [1], dataType: .int32)
        melLen[0] = NSNumber(value: config.totalMelFrames)

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: inputMel),
            "mel_length": MLFeatureValue(multiArray: melLen),
            "cache_channel": MLFeatureValue(multiArray: cacheChannel),
            "cache_time": MLFeatureValue(multiArray: cacheTime),
            "cache_len": MLFeatureValue(multiArray: cacheLen),
        ])

        let encoderOutput = try await encoder.prediction(from: encoderInput)

        // Update encoder cache states using EncoderCacheManager
        let updatedCaches = EncoderCacheManager.extractCachesFromOutput(encoderOutput)
        if let newChannel = updatedCaches.channel {
            self.cacheChannel = newChannel
        }
        if let newTime = updatedCaches.time {
            self.cacheTime = newTime
        }
        if let newLen = updatedCaches.len {
            self.cacheLen = newLen
        }

        guard let encoded = encoderOutput.featureValue(for: "encoded")?.multiArrayValue else {
            throw ASRError.processingFailed("Encoder failed to produce output")
        }

        // Save mel cache for next chunk (last 9 frames)
        melCache = try extractMelCache(from: chunkMel)

        lastEncoderOutput = encoded
        let spot = samples.count / 2
        if spot < samples.count && abs(samples[spot]) > 1e-6 {
            lastSpeechEncoderOutput = encoded
            lastSpeechFrameIndex = absoluteFrameBase + encoded.shape[2].intValue - 1
        }

        // 4. RNNT decode loop for each encoder frame
        let numEncoderFrames = encoded.shape[2].intValue
        var newTokens: [Int] = []

        for t in 0..<numEncoderFrames {
            let encStep = try extractEncoderStep(from: encoded, timeIndex: t)

            // Greedy decode loop (max 10 symbols per frame)
            for _ in 0..<10 {
                let step = try await decodeStep(token: currentToken, h: currentH, c: currentC, encoderStep: encStep)
                let predToken = findMaxIndex(step.logits)
                let stepH = step.h
                let stepC = step.c

                if predToken == config.blankIdx {
                    // Blank token - move to next encoder frame
                    break
                } else {
                    // Non-blank token - emit and update local state
                    newTokens.append(predToken)
                    recordRecognitionToken(predToken, logits: step.logits, frame: absoluteFrameBase + t)
                    currentToken = Int32(predToken)
                    currentH = stepH
                    currentC = stepC
                }
            }
        }

        // Advance the absolute encoder-frame base by this chunk's frame count so
        // the next chunk's token timings continue on the same timeline.
        absoluteFrameBase += numEncoderFrames

        // Save final decoder state back to actor properties atomically
        self.lastToken = currentToken
        self.hState = currentH
        self.cState = currentC

        // Invoke partial callback if new tokens were decoded
        if !newTokens.isEmpty, let callback = partialCallback, let tokenizer = tokenizer {
            let partial = tokenizer.decode(ids: accumulatedTokenIds)
            callback(partial)
        }

        if !newTokens.isEmpty { emitRecognitionSnapshot(isFinal: false) }
        processedChunks += 1
    }

    // MARK: - Encoder Health Probe

    /// Run one encoder prediction with a non-zero mel probe and report whether
    /// the encoder produced any non-zero output.
    ///
    /// On iPadOS cold starts the int8 encoder's ANE `main` entry point can fail
    /// to instantiate (logged by CoreML as
    /// `ANEProgramProcessRequestDirect() Failed with status=0x12`). When that
    /// happens `prediction` does not throw — it silently returns an all-zero
    /// `encoded` buffer, so the RNN-T loop only ever sees blanks and the final
    /// transcript is empty with no error surfaced (issue #739). A single
    /// non-zero probe distinguishes a working encoder (LayerNorm/bias guarantee
    /// non-zero output for non-zero input) from a stillborn ANE program, letting
    /// `loadModels` fail loudly instead of returning empty transcripts.
    ///
    /// Uses throwaway local inputs and does not write the encoder's updated
    /// caches back, so the freshly reset session state is left untouched. The
    /// probe doubles as a model warm-up.
    internal func encoderProducesNonZeroOutput() async throws -> Bool {
        guard let encoder = encoder,
            let cacheChannel = cacheChannel,
            let cacheTime = cacheTime,
            let cacheLen = cacheLen
        else {
            throw ASRError.notInitialized
        }

        // Non-zero mel input ([1, melFeatures, totalMelFrames]) so a healthy
        // encoder is guaranteed to emit non-zero output. A small ramp avoids a
        // degenerate constant that could in theory cancel out.
        let mel = try MLMultiArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: config.totalMelFrames)],
            dataType: .float32
        )
        let melPtr = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)
        for i in 0..<mel.count {
            melPtr[i] = Float(i % 17) * 0.01 + 0.1
        }

        let melLen = try MLMultiArray(shape: [1], dataType: .int32)
        melLen[0] = NSNumber(value: config.totalMelFrames)

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "mel_length": MLFeatureValue(multiArray: melLen),
            "cache_channel": MLFeatureValue(multiArray: cacheChannel),
            "cache_time": MLFeatureValue(multiArray: cacheTime),
            "cache_len": MLFeatureValue(multiArray: cacheLen),
        ])

        let encoderOutput = try await encoder.prediction(from: encoderInput)
        guard let encoded = encoderOutput.featureValue(for: "encoded")?.multiArrayValue else {
            throw ASRError.processingFailed("Encoder probe produced no `encoded` output")
        }

        let outPtr = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
        for i in 0..<encoded.count where outPtr[i] != 0 {
            return true
        }
        return false
    }

    // MARK: - Tensor Utilities

    internal func prependMelCache(to chunkMel: MLMultiArray) throws -> MLMultiArray {
        // Prepend cached mel frames (9) to current chunk mel (112) → [1, 128, 121]
        // Input: chunkMel [1, 128, ~112]
        // Output: [1, 128, 121] = 9 cache + 112 chunk (or padded)

        let chunkFrames = chunkMel.shape[2].intValue
        let totalFrames = config.totalMelFrames

        let result = try MLMultiArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: totalFrames)],
            dataType: .float32
        )
        result.reset(to: 0)

        let resultPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)
        let chunkPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)

        let resultStride0 = result.strides[0].intValue
        let resultStride1 = result.strides[1].intValue
        let resultStride2 = result.strides[2].intValue
        let chunkStride0 = chunkMel.strides[0].intValue
        let chunkStride1 = chunkMel.strides[1].intValue
        let chunkStride2 = chunkMel.strides[2].intValue

        // Copy mel cache (or zeros if first chunk)
        if let melCache = melCache {
            let cachePtr = melCache.dataPointer.bindMemory(to: Float.self, capacity: melCache.count)
            let cacheFrames = melCache.shape[2].intValue
            let cacheStride0 = melCache.strides[0].intValue
            let cacheStride1 = melCache.strides[1].intValue
            let cacheStride2 = melCache.strides[2].intValue

            for mel in 0..<config.melFeatures {
                for t in 0..<cacheFrames {
                    let srcIdx = 0 * cacheStride0 + mel * cacheStride1 + t * cacheStride2
                    let dstIdx = 0 * resultStride0 + mel * resultStride1 + t * resultStride2
                    resultPtr[dstIdx] = cachePtr[srcIdx]
                }
            }
        }

        // Copy chunk mel (after cache position)
        let copyFrames = min(chunkFrames, totalFrames - config.preEncodeCache)
        for mel in 0..<config.melFeatures {
            for t in 0..<copyFrames {
                let srcIdx = 0 * chunkStride0 + mel * chunkStride1 + t * chunkStride2
                let dstIdx = 0 * resultStride0 + mel * resultStride1 + (config.preEncodeCache + t) * resultStride2
                resultPtr[dstIdx] = chunkPtr[srcIdx]
            }
        }

        return result
    }

    internal func extractMelCache(from chunkMel: MLMultiArray) throws -> MLMultiArray {
        // Extract last preEncodeCache (9) frames from chunk mel
        let chunkFrames = chunkMel.shape[2].intValue
        let cacheFrames = min(config.preEncodeCache, chunkFrames)

        let cache = try MLMultiArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: cacheFrames)],
            dataType: .float32
        )

        let srcPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)
        let dstPtr = cache.dataPointer.bindMemory(to: Float.self, capacity: cache.count)

        let srcStride0 = chunkMel.strides[0].intValue
        let srcStride1 = chunkMel.strides[1].intValue
        let srcStride2 = chunkMel.strides[2].intValue
        let dstStride0 = cache.strides[0].intValue
        let dstStride1 = cache.strides[1].intValue
        let dstStride2 = cache.strides[2].intValue

        let startT = chunkFrames - cacheFrames

        for mel in 0..<config.melFeatures {
            for t in 0..<cacheFrames {
                let srcIdx = 0 * srcStride0 + mel * srcStride1 + (startT + t) * srcStride2
                let dstIdx = 0 * dstStride0 + mel * dstStride1 + t * dstStride2
                dstPtr[dstIdx] = srcPtr[srcIdx]
            }
        }

        return cache
    }

    internal func extractEncoderStep(from encoded: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
        // encoded: [1, 1024, T] -> step: [1, 1024, 1]
        let dim = encoded.shape[1].intValue
        let step = try MLMultiArray(shape: [1, NSNumber(value: dim), 1], dataType: .float32)

        let srcPtr = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
        let dstPtr = step.dataPointer.bindMemory(to: Float.self, capacity: step.count)

        let stride0 = encoded.strides[0].intValue
        let stride1 = encoded.strides[1].intValue
        let stride2 = encoded.strides[2].intValue

        for c in 0..<dim {
            let srcIdx = 0 * stride0 + c * stride1 + timeIndex * stride2
            dstPtr[c] = srcPtr[srcIdx]
        }

        return step
    }

    internal func sliceDecoderOutput(_ decoderOut: MLMultiArray) throws -> MLMultiArray {
        // decoder_out: [1, hidden, T] -> [1, hidden, 1] (first frame, index 0)
        // Python uses decoder_out[:, :, :1] which is the FIRST frame
        let hidden = decoderOut.shape[1].intValue

        let result = try MLMultiArray(shape: [1, NSNumber(value: hidden), 1], dataType: .float32)

        let srcPtr = decoderOut.dataPointer.bindMemory(to: Float.self, capacity: decoderOut.count)
        let dstPtr = result.dataPointer.bindMemory(to: Float.self, capacity: result.count)

        let stride0 = decoderOut.strides[0].intValue
        let stride1 = decoderOut.strides[1].intValue
        let stride2 = decoderOut.strides[2].intValue

        // Use FIRST frame (index 0), not last frame
        let firstT = 0
        for c in 0..<hidden {
            let srcIdx = 0 * stride0 + c * stride1 + firstT * stride2
            dstPtr[c] = srcPtr[srcIdx]
        }

        return result
    }

    internal func findMaxIndex(_ logits: MLMultiArray) -> Int {
        // logits: [1, 1, 1, vocab_size+1]
        // Use actual logits count to prevent out-of-bounds when config is incorrect
        let count = logits.count

        guard logits.dataType == .float32, logits.strides.last?.intValue == 1 else {
            return (0..<count).max { logits[$0].floatValue < logits[$1].floatValue } ?? 0
        }
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: count)

        // Use Accelerate framework for vectorized maximum index search
        var maxVal: Float = -Float.infinity
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &maxIdx, vDSP_Length(count))

        return Int(maxIdx)
    }
}

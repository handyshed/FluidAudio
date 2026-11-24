@preconcurrency import CoreML
import Foundation

/// Authoritative stable prefix plus optional revisable tentative suffix for live streaming captions.
public struct NemotronRecognitionDisplay: Sendable, Equatable {
    public let stableText: String
    public let tentativeText: String
    public let committedRevision: Int
    public let audioThroughSample: Int

    public init(
        stableText: String,
        tentativeText: String,
        committedRevision: Int,
        audioThroughSample: Int
    ) {
        self.stableText = stableText
        self.tentativeText = tentativeText
        self.committedRevision = committedRevision
        self.audioThroughSample = audioThroughSample
    }
}

extension StreamingNemotronAsrManager {
    /// Current committed display snapshot split at the last SentencePiece word boundary so an
    /// unfinished trailing word remains tentative until a following word boundary or `finish()`.
    public func recognitionDisplay() throws -> NemotronRecognitionDisplay {
        guard let tokenizer else {
            throw ASRError.notInitialized
        }
        let (stable, tentative) = try Self.splitDisplay(
            committedTokens: accumulatedTokenIds,
            displayTokens: accumulatedTokenIds,
            tokenizer: tokenizer
        )
        return NemotronRecognitionDisplay(
            stableText: stable,
            tentativeText: tentative,
            committedRevision: processedChunks,
            audioThroughSample: processedChunks * config.chunkSamples + audioBuffer.count
        )
    }

    /// Speculatively decodes the buffered audio remainder (zero-padded to a full chunk) using
    /// the already-loaded streaming models without mutating any authoritative streaming state.
    ///
    /// Returns `nil` if the buffered remainder is smaller than `minRemainderSamples`, if a full
    /// chunk is already buffered, or if authoritative state advanced while decoding was in flight.
    public func tentativeDisplay(minRemainderSamples: Int = 3200) async throws -> NemotronRecognitionDisplay? {
        guard let melExtractor,
            let encoder,
            let tokenizer,
            let cacheChannel,
            let cacheTime,
            let cacheLen,
            var currentH = hState,
            var currentC = cState
        else {
            throw ASRError.notInitialized
        }
        guard decoderJoint != nil || (decoder != nil && joint != nil) else {
            throw ASRError.notInitialized
        }

        let remainder = audioBuffer
        guard remainder.count >= minRemainderSamples, remainder.count < config.chunkSamples else {
            return nil
        }

        let startGeneration = sessionGeneration
        let startChunks = processedChunks
        let committedTokens = accumulatedTokenIds
        let realSamples = startChunks * config.chunkSamples + remainder.count
        var currentToken = lastToken

        // Pad the partial remainder to the encoder's fixed chunk size and build mel synchronously
        // before the first `await` so `melCache` cannot change mid-copy.
        var padded = [Float](repeating: 0, count: config.chunkSamples)
        padded.replaceSubrange(0..<remainder.count, with: remainder)
        let chunkMel = try melExtractor.melSpectrogram(samples: padded)
        let inputMel = try prependMelCache(to: chunkMel)

        let melLen = try MLMultiArray(shape: [1], dataType: .int32)
        melLen[0] = NSNumber(value: config.totalMelFrames)

        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: inputMel),
            "mel_length": MLFeatureValue(multiArray: melLen),
            "cache_channel": MLFeatureValue(multiArray: cacheChannel),
            "cache_time": MLFeatureValue(multiArray: cacheTime),
            "cache_len": MLFeatureValue(multiArray: cacheLen),
        ])

        try Task.checkCancellation()
        let encoderOutput = try await encoder.prediction(from: encoderInput)
        try Task.checkCancellation()

        guard sessionGeneration == startGeneration,
            processedChunks == startChunks,
            accumulatedTokenIds.count == committedTokens.count
        else {
            return nil
        }

        guard let encoded = encoderOutput.featureValue(for: "encoded")?.multiArrayValue else {
            throw ASRError.processingFailed("Encoder failed to produce tentative output")
        }

        let numEncoderFrames = encoded.shape[2].intValue
        var previewTokens = committedTokens

        for t in 0..<numEncoderFrames {
            try Task.checkCancellation()
            let encStep = try extractEncoderStep(from: encoded, timeIndex: t)
            for _ in 0..<10 {
                let step = try await decodeStep(
                    token: currentToken, h: currentH, c: currentC, encoderStep: encStep
                )
                let predToken = findMaxIndex(step.logits)
                if predToken == config.blankIdx {
                    break
                }
                previewTokens.append(predToken)
                currentToken = Int32(predToken)
                currentH = step.h
                currentC = step.c
            }
        }

        try Task.checkCancellation()
        guard sessionGeneration == startGeneration,
            processedChunks == startChunks,
            accumulatedTokenIds.count == committedTokens.count
        else {
            return nil
        }

        let (stable, tentative) = try Self.splitDisplay(
            committedTokens: committedTokens,
            displayTokens: previewTokens,
            tokenizer: tokenizer
        )
        return NemotronRecognitionDisplay(
            stableText: stable,
            tentativeText: tentative,
            committedRevision: startChunks,
            audioThroughSample: realSamples
        )
    }

    /// Splits `displayTokens` into an irrevocable `stable` prefix (ending before the last
    /// SentencePiece word boundary in `committedTokens`) and a revisable `tentative` tail.
    internal static func splitDisplay(
        committedTokens: [Int],
        displayTokens: [Int],
        tokenizer: Tokenizer
    ) throws -> (stable: String, tentative: String) {
        let boundary = committedTokens.lastIndex {
            tokenizer.rawToken(for: $0)?.hasPrefix("\u{2581}") == true
        } ?? 0
        let stable = tokenizer.decode(ids: Array(committedTokens.prefix(boundary)))
        let full = tokenizer.decode(ids: displayTokens)
        guard full.hasPrefix(stable) else {
            throw ASRError.processingFailed("Tokenizer prefix changed during tentative display split")
        }
        return (stable, String(full.dropFirst(stable.count)))
    }
}

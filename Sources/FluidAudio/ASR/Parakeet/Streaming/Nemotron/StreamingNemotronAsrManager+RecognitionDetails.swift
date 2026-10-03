@preconcurrency import CoreML
import Foundation

extension StreamingNemotronAsrManager {
    /// Receive snapshots after token emission and at finalization.
    public func setPartialSnapshotCallback(_ callback: @escaping NemotronSnapshotCallback) {
        snapshotCallback = callback
    }

    /// Disable retaining full logits during warm-up or text-only batch processing.
    public func setSkipLogitSaving(_ skip: Bool) { skipLogitSaving = skip }

    /// Decode a token with SentencePiece word-boundary handling.
    public func decodeToken(_ id: Int) -> String {
        tokenizer?.decode(ids: [id]).trimmingCharacters(in: .whitespaces) ?? "<\(id)>"
    }

    /// Return the original vocabulary piece, including its word-boundary marker.
    public func rawDecodeToken(_ id: Int) -> String {
        tokenizer?.rawToken(for: id) ?? "<\(id)>"
    }

    /// Reset a session while retaining allocated encoder and decoder caches.
    public func resetFast() {
        audioBuffer.removeAll(keepingCapacity: true)
        accumulatedTokenIds.removeAll(keepingCapacity: true)
        accumulatedTokenTimings.removeAll(keepingCapacity: true)
        lastFinishTokenTimings.removeAll(keepingCapacity: true)
        absoluteFrameBase = 0
        processedChunks = 0
        clearRecognitionState()
        cacheChannel?.reset(to: 0)
        cacheTime?.reset(to: 0)
        cacheLen?[0] = 1
        hState?.reset(to: 0)
        cState?.reset(to: 0)
        lastToken = Int32(config.blankIdx)
        melCache = nil
    }

    /// Finish with measured confidences, top-16 alternatives and absolute timestamps.
    /// Stop feeding speech and await all pending feeds before finalization.
    public func finishWithRecognitionDetails() async throws -> NemotronRecognitionDetails {
        guard let tokenizer, melExtractor != nil, encoder != nil, decoder != nil, joint != nil else {
            throw ASRError.notInitialized
        }
        if !audioBuffer.isEmpty {
            var chunk = Array(audioBuffer.prefix(config.chunkSamples))
            audioBuffer.removeAll()
            chunk.append(contentsOf: repeatElement(0, count: max(0, config.chunkSamples - chunk.count)))
            try await processChunk(chunk)
        }
        if let encoded = lastSpeechEncoderOutput ?? lastEncoderOutput {
            try await flushRecognitionDecoder(encoded)
        }
        emitRecognitionSnapshot(isFinal: true)
        lastFinishTokenTimings = accumulatedTokenTimings
        let details = NemotronRecognitionDetails(
            text: tokenizer.decode(ids: accumulatedTokenIds), tokenIds: accumulatedTokenIds,
            confidences: accumulatedTokenTimings.map(\.confidence),
            alternatives: savedRecognitionLogits.map {
                $0.map { NemotronRecognitionLogits.alternatives($0, blank: config.blankIdx) } ?? []
            },
            timestamps: accumulatedTokenTimings.map { Int(($0.startTime * 1000).rounded()) }
        )
        accumulatedTokenIds.removeAll()
        accumulatedTokenTimings.removeAll()
        absoluteFrameBase = 0
        clearRecognitionState()
        return details
    }

    /// Finish after successfully processing a full trailing-silence chunk.
    /// Clears only an entirely zero-valued remainder and retains decoder-state flushing.
    /// Callers without that successful feed must use ordinary finalization.
    public func finishSkippingSilentRemainder() async throws -> NemotronRecognitionDetails {
        if audioBuffer.allSatisfy({ $0 == 0 }) { audioBuffer.removeAll() }
        return try await finishWithRecognitionDetails()
    }

    internal func clearRecognitionState() {
        savedRecognitionLogits.removeAll(keepingCapacity: true)
        snapshotRevision = 0
        lastEncoderOutput = nil
        lastSpeechEncoderOutput = nil
        lastSpeechFrameIndex = 0
    }

    internal func recordRecognitionToken(_ token: Int, logits: MLMultiArray, frame: Int) {
        let values = NemotronRecognitionLogits.values(from: logits)
        let start = Double(frame) * ASRConstants.secondsPerEncoderFrame
        accumulatedTokenIds.append(token)
        accumulatedTokenTimings.append(
            TokenTiming(
                token: tokenizer?.rawToken(for: token) ?? "", tokenId: token,
                startTime: start, endTime: start + ASRConstants.secondsPerEncoderFrame,
                confidence: NemotronRecognitionLogits.confidence(values, token: token)
            ))
        savedRecognitionLogits.append(skipLogitSaving ? nil : values)
    }

    internal func emitRecognitionSnapshot(isFinal: Bool) {
        guard let snapshotCallback, let tokenizer else { return }
        snapshotRevision += 1
        snapshotCallback(
            StreamingTranscriptSnapshot(
                revision: snapshotRevision, text: tokenizer.decode(ids: accumulatedTokenIds),
                tokenIds: accumulatedTokenIds, confidences: accumulatedTokenTimings.map(\.confidence),
                timestampsMs: accumulatedTokenTimings.map { Int(($0.startTime * 1000).rounded()) }, isFinal: isFinal
            ))
    }

    internal func decodeStep(
        token: Int32, h: MLMultiArray, c: MLMultiArray, encoderStep: MLMultiArray
    )
        async throws -> (logits: MLMultiArray, h: MLMultiArray, c: MLMultiArray)
    {
        let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenInput[0] = NSNumber(value: token)
        let tokenLength = try MLMultiArray(shape: [1], dataType: .int32)
        tokenLength[0] = 1
        var input: [String: MLFeatureValue] = [
            "token": MLFeatureValue(multiArray: tokenInput), "token_length": MLFeatureValue(multiArray: tokenLength),
            "h_in": MLFeatureValue(multiArray: h), "c_in": MLFeatureValue(multiArray: c),
        ]
        if let decoderJoint {
            input["encoder"] = MLFeatureValue(multiArray: encoderStep)
            let output = try await decoderJoint.prediction(from: MLDictionaryFeatureProvider(dictionary: input))
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
                let hOut = output.featureValue(for: "h_out")?.multiArrayValue,
                let cOut = output.featureValue(for: "c_out")?.multiArrayValue
            else { throw ASRError.processingFailed("Fused decoder_joint failed") }
            return (logits, hOut, cOut)
        }
        guard let decoder, let joint else { throw ASRError.notInitialized }
        let output = try await decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: input))
        guard let decoded = output.featureValue(for: "decoder_out")?.multiArrayValue,
            let hOut = output.featureValue(for: "h_out")?.multiArrayValue,
            let cOut = output.featureValue(for: "c_out")?.multiArrayValue
        else { throw ASRError.processingFailed("Decoder failed") }
        let jointInput = try MLDictionaryFeatureProvider(dictionary: [
            "encoder": MLFeatureValue(multiArray: encoderStep),
            "decoder": MLFeatureValue(multiArray: try sliceDecoderOutput(decoded)),
        ])
        let jointOutput = try await joint.prediction(from: jointInput)
        guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
            throw ASRError.processingFailed("Joint failed")
        }
        return (logits, hOut, cOut)
    }

    private func flushRecognitionDecoder(_ encoded: MLMultiArray) async throws {
        guard var currentH = hState, var currentC = cState else { return }
        let frames = encoded.shape[2].intValue
        guard frames > 0 else { return }
        let encoderStep = try extractEncoderStep(from: encoded, timeIndex: frames - 1)
        var blanks = 0
        for _ in 0..<20 {
            let step = try await decodeStep(token: lastToken, h: currentH, c: currentC, encoderStep: encoderStep)
            let token = findMaxIndex(step.logits)
            if token == config.blankIdx {
                blanks += 1
                if blanks >= 3 { break }
                continue
            }
            recordRecognitionToken(token, logits: step.logits, frame: lastSpeechFrameIndex)
            lastToken = Int32(token)
            currentH = step.h
            currentC = step.c
            blanks = 0
        }
        hState = currentH
        cState = currentC
    }
}

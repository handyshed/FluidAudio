import AVFoundation
@preconcurrency import CoreML
import Foundation

/// Chunk size variant for Nemotron streaming
public enum NemotronChunkSize: Int, Sendable, CaseIterable {
    case ms1120 = 1120  // 1.12s - original
    case ms560 = 560  // 0.56s
    case ms160 = 160  // 0.16s
    case ms80 = 80  // 0.08s

    public var repo: Repo {
        switch self {
        case .ms1120: return .nemotronStreaming1120
        case .ms560: return .nemotronStreaming560
        case .ms160: return .nemotronStreaming160
        case .ms80: return .nemotronStreaming80
        }
    }

    public var folderName: String {
        "nemotron_coreml_\(rawValue)ms"
    }
}

/// Encoder file name for Nemotron streaming (int8 quantized only)
public enum NemotronEncoder {
    static let fileName = "encoder_int8.mlmodelc"
}

/// Configuration for Nemotron Speech Streaming 0.6B
/// Loaded from metadata.json for each chunk size variant
public struct NemotronStreamingConfig: Sendable {
    /// Sample rate in Hz
    public let sampleRate: Int
    /// Number of mel spectrogram features
    public let melFeatures: Int
    /// Mel frames per chunk
    public let chunkMelFrames: Int
    /// Chunk duration in milliseconds
    public let chunkMs: Int
    /// Pre-encode cache size in mel frames (for encoder context)
    public let preEncodeCache: Int
    /// Total mel frames for encoder input (cache + chunk)
    public let totalMelFrames: Int
    /// Vocabulary size
    public let vocabSize: Int
    /// Blank token index (== vocab_size)
    public let blankIdx: Int
    /// Encoder output dimension
    public let encoderDim: Int
    /// Decoder hidden size
    public let decoderHidden: Int
    /// Number of decoder LSTM layers
    public let decoderLayers: Int
    /// Encoder cache shapes
    public let cacheChannelShape: [Int]
    public let cacheTimeShape: [Int]

    /// Audio samples per chunk
    public var chunkSamples: Int { chunkMelFrames * 160 }

    /// Default config for 1120ms chunks (backward compatibility)
    public init() {
        self.sampleRate = 16000
        self.melFeatures = 128
        self.chunkMelFrames = 112
        self.chunkMs = 1120
        self.preEncodeCache = 9
        self.totalMelFrames = 121
        self.vocabSize = 1024
        self.blankIdx = 1024
        self.encoderDim = 1024
        self.decoderHidden = 640
        self.decoderLayers = 2
        self.cacheChannelShape = [1, 24, 70, 1024]
        self.cacheTimeShape = [1, 24, 1024, 8]
    }

    /// Load config from metadata.json
    public init(from metadataURL: URL) throws {
        let data = try Data(contentsOf: metadataURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        self.sampleRate = json["sample_rate"] as? Int ?? 16000
        self.melFeatures = json["mel_features"] as? Int ?? 128
        self.chunkMelFrames = json["chunk_mel_frames"] as? Int ?? 112
        self.chunkMs = json["chunk_ms"] as? Int ?? 1120
        self.preEncodeCache = json["pre_encode_cache"] as? Int ?? 9
        self.totalMelFrames = json["total_mel_frames"] as? Int ?? 121
        self.vocabSize = json["vocab_size"] as? Int ?? 1024
        self.blankIdx = json["blank_idx"] as? Int ?? 1024
        self.encoderDim = json["encoder_dim"] as? Int ?? 1024
        self.decoderHidden = json["decoder_hidden"] as? Int ?? 640
        self.decoderLayers = json["decoder_layers"] as? Int ?? 2
        self.cacheChannelShape = json["cache_channel_shape"] as? [Int] ?? [1, 24, 70, 1024]
        self.cacheTimeShape = json["cache_time_shape"] as? [Int] ?? [1, 24, 1024, 8]
    }
}

/// Callback invoked when new tokens are decoded (for live transcription updates)
public typealias NemotronPartialCallback = @Sendable (String) -> Void

/// High-level manager for Nemotron Speech Streaming 0.6B pipeline.
/// A candidate token from the RNNT joint network with its softmax probability.
public struct TokenCandidate: Sendable {
    public let tokenId: Int
    public let probability: Float

    public init(tokenId: Int, probability: Float) {
        self.tokenId = tokenId
        self.probability = probability
    }
}

/// Incremental transcript snapshot emitted during streaming.
/// Used for preview-only work (e.g., speculative punctuation) while speech continues.
/// Final deterministic output still comes from `finish()`.
public struct StreamingTranscriptSnapshot: Sendable {
    /// Monotonically increasing revision number. Host can ignore stale updates.
    public let revision: Int
    /// Current accumulated transcript text.
    public let text: String
    /// Raw token IDs (SentencePiece).
    public let tokenIds: [Int]
    /// Per-token confidence scores.
    public let confidences: [Float]
    /// Per-token timestamps in Nemotron frame units (~8ms each).
    public let timestampsMs: [Int]
    /// Always false for partial snapshots. Final output comes from finish().
    public let isFinal: Bool
}

public typealias NemotronSnapshotCallback = @Sendable (StreamingTranscriptSnapshot) -> Void

/// Implements true streaming with encoder cache states.
public actor NemotronStreamingAsrManager {
    private let logger = AppLogger(category: "NemotronStreaming")

    // Models
    private var preprocessor: MLModel?
    private var encoder: MLModel?
    private var decoder: MLModel?
    private var joint: MLModel?

    // Components
    private let audioConverter = AudioConverter()
    private var tokenizer: Tokenizer?

    // Configuration (loaded from metadata.json)
    public private(set) var config: NemotronStreamingConfig

    // Audio Buffer
    private var audioBuffer: [Float] = []

    // Accumulated token IDs and per-token confidences
    private var accumulatedTokenIds: [Int] = []
    private var accumulatedTokenTimestamps: [Int] = []  // ms timestamp for each token
    private var totalMelFramesProcessed: Int = 0
    private var accumulatedConfidences: [Float] = []
    // N-best alternatives per emitted token position
    private var accumulatedAlternatives: [[TokenCandidate]] = []
    // Raw logits saved during streaming for deferred top-K extraction
    private var savedLogits: [[Float]] = []

    // Encoder cache states
    private var cacheChannel: MLMultiArray?
    private var cacheTime: MLMultiArray?
    private var cacheLen: MLMultiArray?

    // Mel cache (last 9 frames from previous chunk)
    private var melCache: MLMultiArray?

    // Decoder LSTM states
    private var hState: MLMultiArray?
    private var cState: MLMultiArray?
    private var lastToken: Int32

    // Callbacks
    private var partialCallback: NemotronPartialCallback?
    private var snapshotCallback: NemotronSnapshotCallback?
    private var snapshotRevision: Int = 0

    // Last encoder output (for decoder flush on finish)
    private var lastEncoderOutput: MLMultiArray?
    // Last encoder output from actual speech (not silence) — used for flush
    private var lastSpeechEncoderOutput: MLMultiArray?

    // Stats
    private var processedChunks: Int = 0

    public private(set) var mlConfiguration: MLModelConfiguration

    public init(configuration: MLModelConfiguration = MLModelConfiguration()) {
        self.mlConfiguration = configuration
        self.config = NemotronStreamingConfig()
        self.lastToken = Int32(config.blankIdx)
    }

    /// Set callback for partial transcription updates (text only).
    public func setPartialCallback(_ callback: @escaping NemotronPartialCallback) {
        self.partialCallback = callback
    }

    /// Set callback for structured incremental snapshots.
    /// Emitted whenever new tokens are decoded. Includes text, confidences, timestamps.
    /// For preview-only work — final output still comes from finish().
    public func setPartialSnapshotCallback(_ callback: @escaping NemotronSnapshotCallback) {
        self.snapshotCallback = callback
    }

    /// Load models from a directory containing preprocessor, encoder, decoder, joint, and tokenizer
    /// - Parameters:
    ///   - modelDir: Directory containing the model files
    public func loadModels(modelDir: URL) async throws {
        logger.info("Loading Nemotron CoreML models from \(modelDir.path)...")

        // Load config from metadata.json
        let metadataPath = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.metadata)
        if FileManager.default.fileExists(atPath: metadataPath.path) {
            self.config = try NemotronStreamingConfig(from: metadataPath)
            logger.info("Loaded config: \(config.chunkMs)ms chunks, \(config.chunkMelFrames) mel frames")
        }

        // Load preprocessor
        let preprocessorPath = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.preprocessorFile)
        self.preprocessor = try await MLModel.load(contentsOf: preprocessorPath, configuration: mlConfiguration)

        // Load encoder (int8 quantized)
        let encoderPath = modelDir.appendingPathComponent("encoder").appendingPathComponent(NemotronEncoder.fileName)
        self.encoder = try await MLModel.load(contentsOf: encoderPath, configuration: mlConfiguration)

        // Load decoder
        let decoderPath = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.decoderFile)
        self.decoder = try await MLModel.load(contentsOf: decoderPath, configuration: mlConfiguration)

        // Load joint
        let jointPath = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.jointFile)
        self.joint = try await MLModel.load(contentsOf: jointPath, configuration: mlConfiguration)

        // Load tokenizer
        let tokenizerUrl = modelDir.appendingPathComponent(ModelNames.NemotronStreaming.tokenizer)
        self.tokenizer = try Tokenizer(vocabPath: tokenizerUrl)

        // Initialize states
        try resetStates()

        logger.info("Nemotron models loaded successfully (\(config.chunkMs)ms chunks).")
    }

    /// Reset all states for a new transcription session.
    public func reset() async {
        audioBuffer.removeAll()
        accumulatedTokenIds.removeAll()
        accumulatedConfidences.removeAll()
        accumulatedAlternatives.removeAll()
        accumulatedTokenTimestamps.removeAll()
        totalMelFramesProcessed = 0
        savedLogits.removeAll()
        lastEncoderOutput = nil
        lastSpeechEncoderOutput = nil
        processedChunks = 0
        snapshotRevision = 0
        try? resetStates()
    }

    /// Fast reset for batch processing — zeros existing buffers without reallocating.
    public func resetFast() {
        audioBuffer.removeAll(keepingCapacity: true)
        accumulatedTokenIds.removeAll(keepingCapacity: true)
        accumulatedConfidences.removeAll(keepingCapacity: true)
        accumulatedAlternatives.removeAll(keepingCapacity: true)
        accumulatedTokenTimestamps.removeAll(keepingCapacity: true)
        totalMelFramesProcessed = 0
        savedLogits.removeAll(keepingCapacity: true)
        lastEncoderOutput = nil
        lastSpeechEncoderOutput = nil
        melCache = nil
        processedChunks = 0
        lastToken = Int32(config.blankIdx)

        // Zero existing tensors instead of reallocating
        if let c = cacheChannel { c.reset(to: 0) }
        if let t = cacheTime { t.reset(to: 0) }
        if let l = cacheLen { l[0] = 0 }
        if let h = hState { h.reset(to: 0) }
        if let c = cState { c.reset(to: 0) }
    }

    private func resetStates() throws {
        // Encoder cache states
        cacheChannel = try MLMultiArray(
            shape: config.cacheChannelShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        cacheChannel?.reset(to: 0)

        cacheTime = try MLMultiArray(
            shape: config.cacheTimeShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        cacheTime?.reset(to: 0)

        cacheLen = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLen?[0] = 0

        // Mel cache (will be initialized on first chunk)
        melCache = nil

        // Decoder LSTM states
        hState = try MLMultiArray(
            shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
            dataType: .float32
        )
        hState?.reset(to: 0)

        cState = try MLMultiArray(
            shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
            dataType: .float32
        )
        cState?.reset(to: 0)

        lastToken = Int32(config.blankIdx)
    }

    /// Append audio buffer for processing
    public func appendAudio(_ buffer: AVAudioPCMBuffer) throws {
        let samples = try audioConverter.resampleBuffer(buffer)
        audioBuffer.append(contentsOf: samples)
    }

    /// Process audio and return partial transcript
    public func process(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        let samples = try audioConverter.resampleBuffer(audioBuffer)
        return try await processSamples(samples)
    }

    /// Skip saving logits for faster batch processing (no top-K extraction in finish).
    private var skipLogitSaving = false

    public func setSkipLogitSaving(_ skip: Bool) {
        skipLogitSaving = skip
    }

    /// Process raw 16kHz Float32 mono samples directly — skips resampling.
    /// Use when audio is already in the correct format.
    public func processSamples(_ samples: [Float]) async throws -> String {
        self.audioBuffer.append(contentsOf: samples)

        // Process complete chunks. We pop each chunk off the buffer BEFORE the
        // `await processChunk(chunk)` call — not after — so the buffer mutation
        // is atomic relative to actor reentrancy. If a concurrent processSamples
        // call enters the actor while processChunk is awaiting, it sees the
        // already-advanced buffer and operates on fresh samples instead of
        // racing for the same prefix.
        //
        // The previous order (prefix → await → removeFirst) crashed with
        // "Can't remove more items from a collection than it has" when two
        // concurrent processSamples calls overlapped: both saw the same
        // prefix, both awaited processChunk, both resumed and called
        // removeFirst(chunkSamples), one of them underflowing the buffer.
        while self.audioBuffer.count >= config.chunkSamples {
            let chunk = Array(self.audioBuffer.prefix(config.chunkSamples))
            self.audioBuffer.removeFirst(config.chunkSamples)
            try await processChunk(chunk)
        }

        return ""
    }

    /// Finish processing and return final transcript with per-token confidences.
    /// Each confidence is the softmax probability of the chosen token from the joint network logits.
    public func finish() async throws -> (text: String, confidences: [Float], alternatives: [[TokenCandidate]], timestamps: [Int]) {
        // Process remaining audio (padded if needed). Same reentrancy discipline
        // as processSamples: drain the buffer BEFORE the await so a concurrent
        // call can't see stale samples while processChunk is suspended.
        if !audioBuffer.isEmpty {
            let paddingNeeded = config.chunkSamples - audioBuffer.count
            if paddingNeeded > 0 {
                audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
            }

            let chunk = Array(audioBuffer.prefix(config.chunkSamples))
            audioBuffer.removeAll()
            try await processChunk(chunk)
        }

        // Flush any tokens buffered in the decoder LSTM state.
        // Use the last SPEECH encoder output (not silence) — the joint network
        // needs speech context to emit remaining tokens. If silence was fed
        // after speech, lastEncoderOutput would be from silence and the joint
        // network would output blanks, losing the final words.
        let flushEncoder = lastSpeechEncoderOutput ?? lastEncoderOutput
        if let encoded = flushEncoder {
            try await flushDecoderState(lastEncoderOutput: encoded)
        }

        // Extract top-K alternatives from saved logits (deferred from streaming)
        let k = 16
        for logitArray in savedLogits {
            let vocabSize = logitArray.count
            var maxVal = logitArray[0]
            for i in 1..<vocabSize { if logitArray[i] > maxVal { maxVal = logitArray[i] } }
            var sumExp: Float = 0
            for i in 0..<vocabSize { sumExp += exp(logitArray[i] - maxVal) }

            var candidates: [(Int, Float)] = []
            for i in 0..<vocabSize {
                let prob = exp(logitArray[i] - maxVal) / sumExp
                if candidates.count < k {
                    candidates.append((i, prob))
                    if candidates.count == k { candidates.sort { $0.1 > $1.1 } }
                } else if prob > candidates[k - 1].1 {
                    candidates[k - 1] = (i, prob)
                    candidates.sort { $0.1 > $1.1 }
                }
            }
            if candidates.count < k { candidates.sort { $0.1 > $1.1 } }
            accumulatedAlternatives.append(
                candidates.filter { $0.0 != config.blankIdx }
                    .map { TokenCandidate(tokenId: $0.0, probability: $0.1) }
            )
        }

        // Decode accumulated tokens
        guard let tokenizer = tokenizer else { return ("", [], [] as [[TokenCandidate]], []) }
        let transcript = tokenizer.decode(ids: accumulatedTokenIds)
        let confidences = accumulatedConfidences
        let alternatives = accumulatedAlternatives
        let timestamps = accumulatedTokenTimestamps
        accumulatedTokenIds.removeAll()
        accumulatedConfidences.removeAll()
        accumulatedAlternatives.removeAll()
        accumulatedTokenTimestamps.removeAll()
        totalMelFramesProcessed = 0
        savedLogits.removeAll()

        return (transcript, confidences, alternatives, timestamps)
    }

    /// Decode a single token ID to its string representation.
    public func decodeToken(_ id: Int) -> String {
        tokenizer?.decode(ids: [id]).trimmingCharacters(in: .whitespaces) ?? "<\(id)>"
    }

    /// Decode a single token ID preserving raw SentencePiece representation (▁ prefix = word boundary).
    public func rawDecodeToken(_ id: Int) -> String {
        tokenizer?.rawToken(id: id) ?? "<\(id)>"
    }

    /// Flush buffered tokens from decoder LSTM state after the final encoder frame.
    /// Runs additional decoder iterations using the last encoder frame to drain any
    /// tokens that are pending in the LSTM hidden state but haven't been emitted
    /// because the greedy decode loop broke on a blank token.
    private func flushDecoderState(lastEncoderOutput encoded: MLMultiArray) async throws {
        guard let decoder = decoder,
            let joint = joint,
            var currentH = hState,
            var currentC = cState
        else {
            return
        }

        // Use the last encoder frame for all flush iterations
        let numFrames = encoded.shape[2].intValue
        guard numFrames > 0 else { return }
        let encStep = try extractEncoderStep(from: encoded, timeIndex: numFrames - 1)

        var consecutiveBlanks = 0
        let maxFlushIterations = 20

        for _ in 0..<maxFlushIterations {
            let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
            tokenInput[0] = NSNumber(value: lastToken)

            let tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
            tokenLen[0] = 1

            let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "token": MLFeatureValue(multiArray: tokenInput),
                "token_length": MLFeatureValue(multiArray: tokenLen),
                "h_in": MLFeatureValue(multiArray: currentH),
                "c_in": MLFeatureValue(multiArray: currentC),
            ])

            let decoderOutput = try await decoder.prediction(from: decoderInput)

            guard let decoderOut = decoderOutput.featureValue(for: "decoder_out")?.multiArrayValue,
                let hOut = decoderOutput.featureValue(for: "h_out")?.multiArrayValue,
                let cOut = decoderOutput.featureValue(for: "c_out")?.multiArrayValue
            else {
                return
            }

            let decoderStep = try sliceDecoderOutput(decoderOut)

            let jointInput = try MLDictionaryFeatureProvider(dictionary: [
                "encoder": MLFeatureValue(multiArray: encStep),
                "decoder": MLFeatureValue(multiArray: decoderStep),
            ])

            let jointOutput = try await joint.prediction(from: jointInput)

            guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                return
            }

            let (predToken, confidence) = argmaxWithConfidence(logits)

            if predToken == config.blankIdx {
                consecutiveBlanks += 1
                if consecutiveBlanks >= 3 {
                    break
                }
            } else {
                accumulatedTokenIds.append(predToken)
                accumulatedConfidences.append(confidence)
                lastToken = Int32(predToken)
                currentH = hOut
                currentC = cOut
                consecutiveBlanks = 0
            }
        }

        self.hState = currentH
        self.cState = currentC
    }

    /// Get current partial transcript without finishing
    public func getPartialTranscript() -> String {
        guard let tokenizer = tokenizer else { return "" }
        return tokenizer.decode(ids: accumulatedTokenIds)
    }

    /// Process a single audio chunk through the full pipeline
    private func processChunk(_ samples: [Float]) async throws {
        guard let preprocessor = preprocessor,
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

        // 1. Preprocessor: audio -> mel spectrogram
        let audioArray = try createAudioArray(samples)
        let audioLen = try MLMultiArray(shape: [1], dataType: .int32)
        audioLen[0] = NSNumber(value: samples.count)

        let preprocInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: audioArray),
            "audio_length": MLFeatureValue(multiArray: audioLen),
        ])

        let preprocOutput = try await preprocessor.prediction(from: preprocInput)
        guard let chunkMel = preprocOutput.featureValue(for: "mel")?.multiArrayValue else {
            throw ASRError.processingFailed("Preprocessor failed to produce mel output")
        }

        // 2. Build encoder input: prepend mel_cache (9 frames) + current chunk mel
        let inputMel = try buildEncoderMel(chunkMel: chunkMel)

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

        // Update encoder cache states
        if let newCacheChannel = encoderOutput.featureValue(for: "cache_channel_out")?.multiArrayValue {
            self.cacheChannel = newCacheChannel
        }
        if let newCacheTime = encoderOutput.featureValue(for: "cache_time_out")?.multiArrayValue {
            self.cacheTime = newCacheTime
        }
        if let newCacheLen = encoderOutput.featureValue(for: "cache_len_out")?.multiArrayValue {
            self.cacheLen = newCacheLen
        }

        guard let encoded = encoderOutput.featureValue(for: "encoded")?.multiArrayValue else {
            throw ASRError.processingFailed("Encoder failed to produce output")
        }

        // Save mel cache for next chunk (last 9 frames)
        melCache = try extractMelCache(from: chunkMel)

        // Save encoder output for potential flush on finish
        lastEncoderOutput = encoded

        // Save speech encoder output for flush — skip if chunk is pure silence.
        // Quick check: just test a few samples rather than scanning all 2560.
        let spot = samples.count / 2
        if spot < samples.count && abs(samples[spot]) > 1e-6 {
            lastSpeechEncoderOutput = encoded
        }

        // 4. RNNT decode loop for each encoder frame
        let numEncoderFrames = encoded.shape[2].intValue
        var newTokens: [Int] = []

        for t in 0..<numEncoderFrames {
            let encStep = try extractEncoderStep(from: encoded, timeIndex: t)

            // Greedy decode loop (max 10 symbols per frame)
            for _ in 0..<10 {
                let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
                tokenInput[0] = NSNumber(value: lastToken)

                let tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
                tokenLen[0] = 1

                let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                    "token": MLFeatureValue(multiArray: tokenInput),
                    "token_length": MLFeatureValue(multiArray: tokenLen),
                    "h_in": MLFeatureValue(multiArray: currentH),
                    "c_in": MLFeatureValue(multiArray: currentC),
                ])

                let decoderOutput = try await decoder.prediction(from: decoderInput)

                guard let decoderOut = decoderOutput.featureValue(for: "decoder_out")?.multiArrayValue,
                    let hOut = decoderOutput.featureValue(for: "h_out")?.multiArrayValue,
                    let cOut = decoderOutput.featureValue(for: "c_out")?.multiArrayValue
                else {
                    throw ASRError.processingFailed("Decoder failed")
                }

                // Joint: encoder_step + decoder_out -> logits
                let decoderStep = try sliceDecoderOutput(decoderOut)

                let jointInput = try MLDictionaryFeatureProvider(dictionary: [
                    "encoder": MLFeatureValue(multiArray: encStep),
                    "decoder": MLFeatureValue(multiArray: decoderStep),
                ])

                let jointOutput = try await joint.prediction(from: jointInput)

                guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                    throw ASRError.processingFailed("Joint failed")
                }

                // Argmax + confidence from softmax probability of chosen token
                let (predToken, confidence) = argmaxWithConfidence(logits)

                if predToken == config.blankIdx {
                    // Blank token - move to next encoder frame
                    break
                } else {
                    // Non-blank token - emit and update state
                    newTokens.append(predToken)
                    accumulatedTokenIds.append(predToken)
                    accumulatedConfidences.append(confidence)
                    // Record timestamp: each mel frame = 10ms
                    let timestampMs = (totalMelFramesProcessed + t) * 10
                    accumulatedTokenTimestamps.append(timestampMs)
                    // Save raw logits for deferred top-K extraction (just a memcpy)
                    if !skipLogitSaving {
                        let vocabSize = config.vocabSize + 1
                        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: vocabSize)
                        savedLogits.append(Array(UnsafeBufferPointer(start: ptr, count: vocabSize)))
                    }
                    lastToken = Int32(predToken)
                    // Update local variables for next iteration in this chunk
                    currentH = hOut
                    currentC = cOut
                }
            }
        }

        // Update total mel frames processed (for token timestamps)
        totalMelFramesProcessed += numEncoderFrames

        // Save final decoder state back to actor properties for next chunk
        self.hState = currentH
        self.cState = currentC

        // Invoke partial callbacks if new tokens were decoded
        if !newTokens.isEmpty {
            if let callback = partialCallback, let tokenizer = tokenizer {
                let partial = tokenizer.decode(ids: accumulatedTokenIds)
                callback(partial)
            }
            if let snapshotCb = snapshotCallback, let tokenizer = tokenizer {
                snapshotRevision += 1
                let snapshot = StreamingTranscriptSnapshot(
                    revision: snapshotRevision,
                    text: tokenizer.decode(ids: accumulatedTokenIds),
                    tokenIds: accumulatedTokenIds,
                    confidences: accumulatedConfidences,
                    timestampsMs: accumulatedTokenTimestamps,
                    isFinal: false
                )
                snapshotCb(snapshot)
            }
        }

        processedChunks += 1
    }

    // MARK: - Helper Methods

    private func createAudioArray(_ samples: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
        ptr.update(from: samples, count: samples.count)
        return array
    }

    private func buildEncoderMel(chunkMel: MLMultiArray) throws -> MLMultiArray {
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

        // Copy mel cache (or zeros if first chunk)
        if let melCache = melCache {
            let cachePtr = melCache.dataPointer.bindMemory(to: Float.self, capacity: melCache.count)
            let cacheFrames = melCache.shape[2].intValue

            for mel in 0..<config.melFeatures {
                for t in 0..<cacheFrames {
                    let srcIdx = mel * cacheFrames + t
                    let dstIdx = mel * totalFrames + t
                    resultPtr[dstIdx] = cachePtr[srcIdx]
                }
            }
        }

        // Copy chunk mel (after cache position)
        let copyFrames = min(chunkFrames, totalFrames - config.preEncodeCache)
        for mel in 0..<config.melFeatures {
            for t in 0..<copyFrames {
                let srcIdx = mel * chunkFrames + t
                let dstIdx = mel * totalFrames + (config.preEncodeCache + t)
                resultPtr[dstIdx] = chunkPtr[srcIdx]
            }
        }

        return result
    }

    private func extractMelCache(from chunkMel: MLMultiArray) throws -> MLMultiArray {
        // Extract last preEncodeCache (9) frames from chunk mel
        let chunkFrames = chunkMel.shape[2].intValue
        let cacheFrames = min(config.preEncodeCache, chunkFrames)

        let cache = try MLMultiArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: cacheFrames)],
            dataType: .float32
        )

        let srcPtr = chunkMel.dataPointer.bindMemory(to: Float.self, capacity: chunkMel.count)
        let dstPtr = cache.dataPointer.bindMemory(to: Float.self, capacity: cache.count)

        let startT = chunkFrames - cacheFrames

        for mel in 0..<config.melFeatures {
            for t in 0..<cacheFrames {
                let srcIdx = mel * chunkFrames + (startT + t)
                let dstIdx = mel * cacheFrames + t
                dstPtr[dstIdx] = srcPtr[srcIdx]
            }
        }

        return cache
    }

    private func extractEncoderStep(from encoded: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
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

    private func sliceDecoderOutput(_ decoderOut: MLMultiArray) throws -> MLMultiArray {
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

    /// Argmax over logits, returning (tokenIndex, softmaxProbability).
    /// The probability is the softmax of the chosen token — higher means more confident.
    private func argmaxWithConfidence(_ logits: MLMultiArray) -> (Int, Float) {
        let result = topKWithConfidence(logits, k: 1)
        return (result[0].tokenId, result[0].probability)
    }

    /// Top-K tokens from joint network logits with softmax probabilities.
    /// Returns candidates sorted by probability (highest first).
    /// The blank token is included if it's in the top-K.
    private func topKWithConfidence(_ logits: MLMultiArray, k: Int) -> [TokenCandidate] {
        let vocabSize = config.vocabSize + 1  // includes blank
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)

        // Find max logit for numerically stable softmax
        var maxVal = ptr[0]
        for i in 1..<vocabSize {
            if ptr[i] > maxVal { maxVal = ptr[i] }
        }

        // Compute softmax denominator
        var sumExp: Float = 0
        for i in 0..<vocabSize {
            sumExp += exp(ptr[i] - maxVal)
        }

        // Build (index, probability) pairs and partial sort for top-K
        // For small K, a simple selection is faster than full sort
        var candidates: [(Int, Float)] = []
        candidates.reserveCapacity(k)

        for i in 0..<vocabSize {
            let prob = exp(ptr[i] - maxVal) / sumExp
            if candidates.count < k {
                candidates.append((i, prob))
                if candidates.count == k {
                    candidates.sort { $0.1 > $1.1 }
                }
            } else if prob > candidates[k - 1].1 {
                candidates[k - 1] = (i, prob)
                // Re-sort to maintain order
                candidates.sort { $0.1 > $1.1 }
            }
        }

        if candidates.count < k {
            candidates.sort { $0.1 > $1.1 }
        }

        return candidates.map { TokenCandidate(tokenId: $0.0, probability: $0.1) }
    }
}

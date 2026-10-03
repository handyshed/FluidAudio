import AVFoundation
@preconcurrency import CoreML
import Foundation

/// Compatibility adapter for the fork's original Nemotron API.
/// All inference uses the modern upstream engine, including native mel and optional B1 fusion.
public actor NemotronStreamingAsrManager {
    private let engine: StreamingNemotronAsrManager
    private let audioConverter = AudioConverter()
    public var config: NemotronStreamingConfig { get async { await engine.config } }
    public init(configuration: sending MLModelConfiguration? = nil) {
        engine = StreamingNemotronAsrManager(configuration: configuration, requestedChunkSize: .ms1120)
    }
    public func loadModels(modelDir: URL) async throws { try await engine.loadModels(from: modelDir) }
    public func reset() async { await engine.reset() }
    public func resetFast() async { await engine.resetFast() }
    public func setSkipLogitSaving(_ skip: Bool) async { await engine.setSkipLogitSaving(skip) }
    public func setPartialCallback(_ callback: @escaping NemotronPartialCallback) async {
        await engine.setPartialCallback(callback)
    }
    public func setPartialSnapshotCallback(_ callback: @escaping NemotronSnapshotCallback) async {
        await engine.setPartialSnapshotCallback(callback)
    }
    public func appendAudio(_ buffer: AVAudioPCMBuffer) async throws {
        let samples = try audioConverter.resampleBuffer(buffer)
        await engine.appendSamples(samples)
    }
    public func process(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        try await engine.processSamples(audioConverter.resampleBuffer(audioBuffer))
    }
    public func processSamples(_ samples: [Float]) async throws -> String { try await engine.processSamples(samples) }
    public func getPartialTranscript() async -> String { await engine.getPartialTranscript() }
    public func decodeToken(_ id: Int) async -> String { await engine.decodeToken(id) }
    public func rawDecodeToken(_ id: Int) async -> String { await engine.rawDecodeToken(id) }
    public func finish() async throws -> (
        text: String, confidences: [Float], alternatives: [[TokenCandidate]], timestamps: [Int]
    ) {
        let result = try await engine.finishWithRecognitionDetails()
        return (result.text, result.confidences, result.alternatives, result.timestamps)
    }
    public func finishSkippingSilentRemainder() async throws -> (
        text: String, confidences: [Float], alternatives: [[TokenCandidate]], timestamps: [Int]
    ) {
        let result = try await engine.finishSkippingSilentRemainder()
        return (result.text, result.confidences, result.alternatives, result.timestamps)
    }
}

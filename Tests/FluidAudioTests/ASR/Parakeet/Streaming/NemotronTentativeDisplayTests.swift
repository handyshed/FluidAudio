@preconcurrency import AVFoundation
@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class NemotronTentativeDisplayTests: XCTestCase {

    private func makeTokenizer(_ vocab: [String: String]) throws -> Tokenizer {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nemotron-tok-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("tokenizer.json")
        let data = try JSONSerialization.data(withJSONObject: vocab)
        try data.write(to: file)
        return try Tokenizer(vocabPath: file)
    }

    func testSplitDisplayKeepsLastCommittedWordTentative() throws {
        let tokenizer = try makeTokenizer([
            "1": "\u{2581}Send",
            "2": "\u{2581}it",
            "3": "\u{2581}on",
            "4": "\u{2581}Thurs",
            "5": "day",
        ])

        let committed = [1, 2, 3, 4]
        let committedSplit = try StreamingNemotronAsrManager.splitDisplay(
            committedTokens: committed,
            displayTokens: committed,
            tokenizer: tokenizer
        )
        XCTAssertEqual(committedSplit.stable, "Send it on")
        XCTAssertEqual(committedSplit.tentative, " Thurs")
        XCTAssertEqual(committedSplit.stable + committedSplit.tentative, "Send it on Thurs")

        let previewSplit = try StreamingNemotronAsrManager.splitDisplay(
            committedTokens: committed,
            displayTokens: [1, 2, 3, 4, 5],
            tokenizer: tokenizer
        )
        XCTAssertEqual(previewSplit.stable, "Send it on")
        XCTAssertEqual(previewSplit.tentative, " Thursday")
        XCTAssertEqual(previewSplit.stable + previewSplit.tentative, "Send it on Thursday")
    }

    func testSplitDisplayHandlesEmptyCommittedAndFirstWord() throws {
        let tokenizer = try makeTokenizer([
            "1": "\u{2581}Hello",
            "2": "\u{2581}world",
        ])

        let emptyCommitted = try StreamingNemotronAsrManager.splitDisplay(
            committedTokens: [],
            displayTokens: [1, 2],
            tokenizer: tokenizer
        )
        XCTAssertEqual(emptyCommitted.stable, "")
        XCTAssertEqual(emptyCommitted.tentative, "Hello world")

        let singleCommitted = try StreamingNemotronAsrManager.splitDisplay(
            committedTokens: [1],
            displayTokens: [1, 2],
            tokenizer: tokenizer
        )
        XCTAssertEqual(singleCommitted.stable, "")
        XCTAssertEqual(singleCommitted.tentative, "Hello world")
    }

    func testRealModelSingleCopyTentativeImmutabilityAndParity() async throws {
        let modelDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("FluidAudio/Models", isDirectory: true)
            .appendingPathComponent(NemotronChunkSize.ms1120.repo.folderName, isDirectory: true)
        let encoderPath = modelDir.appendingPathComponent("encoder").appendingPathComponent(NemotronEncoder.fileName)
        guard FileManager.default.fileExists(atPath: encoderPath.path) else {
            throw XCTSkip("Cached Nemotron streaming model not present at \(modelDir.path)")
        }

        // Synthesize a spoken-style 16 kHz clip using macOS `say` into a temp WAV so the model
        // has real multi-word speech to decode across multiple chunks without committing media.
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nemotron-tentative-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        let aiffURL = tmpDir.appendingPathComponent("clip.aiff")
        let wavURL = tmpDir.appendingPathComponent("clip.wav")

        let say = Process()
        say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        say.arguments = ["-o", aiffURL.path, "Please send the quarterly report on Thursday morning."]
        try say.run()
        say.waitUntilExit()
        guard say.terminationStatus == 0 else {
            throw XCTSkip("macOS say command unavailable")
        }

        let convert = Process()
        convert.executableURL = URL(fileURLWithPath: "/usr/bin/afconvert")
        convert.arguments = ["-f", "WAVE", "-d", "LEF32@16000", "-c", "1", aiffURL.path, wavURL.path]
        try convert.run()
        convert.waitUntilExit()
        XCTAssertEqual(convert.terminationStatus, 0)

        let audioFile = try AVAudioFile(forReading: wavURL)
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        let frameCount = AVAudioFrameCount(audioFile.length)
        let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        try audioFile.read(into: pcm)
        let samples = Array(UnsafeBufferPointer(start: pcm.floatChannelData![0], count: Int(pcm.frameLength)))
        XCTAssertGreaterThan(samples.count, 16000)

        let engine = StreamingNemotronAsrManager(requestedChunkSize: .ms1120)
        try await engine.loadModels(from: modelDir)

        func tensorBytes(_ array: MLMultiArray?) -> Data? {
            guard let array else { return nil }
            let elementSize: Int
            switch array.dataType {
            case .double: elementSize = 8
            case .float32, .int32: elementSize = 4
            case .float16: elementSize = 2
            default: elementSize = 4
            }
            let shape = array.shape.map(\.intValue)
            let strides = array.strides.map(\.intValue)
            let span = 1 + zip(shape, strides).reduce(0) { $0 + ($1.0 - 1) * $1.1 }
            return Data(bytes: array.dataPointer, count: span * elementSize)
        }

        func runPass(withTentative: Bool) async throws -> (
            result: NemotronRecognitionDetails,
            tentativeUpdates: Int,
            firstTentativeSample: Int?,
            firstCommittedSample: Int?
        ) {
            await engine.reset()
            let pieceSize = 4800 // 300ms at 16kHz
            var offset = 0
            var tentativeUpdates = 0
            var firstTentativeSample: Int?
            var firstCommittedSample: Int?

            while offset < samples.count {
                let end = min(samples.count, offset + pieceSize)
                let piece = Array(samples[offset..<end])
                offset = end
                _ = try await engine.processSamples(piece)

                let committed = try await engine.recognitionDisplay()
                if firstCommittedSample == nil, !(committed.stableText + committed.tentativeText).isEmpty {
                    firstCommittedSample = committed.audioThroughSample
                }

                if withTentative {
                    let chBefore = await tensorBytes(engine.cacheChannel)
                    let tmBefore = await tensorBytes(engine.cacheTime)
                    let lnBefore = await tensorBytes(engine.cacheLen)
                    let melBefore = await tensorBytes(engine.melCache)
                    let hBefore = await tensorBytes(engine.hState)
                    let cBefore = await tensorBytes(engine.cState)
                    let tokBefore = await engine.accumulatedTokenIds
                    let chunksBefore = await engine.processedChunks
                    let bufBefore = await engine.audioBuffer

                    let preview = try await engine.tentativeDisplay(minRemainderSamples: 3200)

                    // Authoritative state must remain byte-for-byte untouched.
                    let chAfter = await tensorBytes(engine.cacheChannel)
                    let tmAfter = await tensorBytes(engine.cacheTime)
                    let lnAfter = await tensorBytes(engine.cacheLen)
                    let melAfter = await tensorBytes(engine.melCache)
                    let hAfter = await tensorBytes(engine.hState)
                    let cAfter = await tensorBytes(engine.cState)
                    let tokAfter = await engine.accumulatedTokenIds
                    let chunksAfter = await engine.processedChunks
                    let bufAfter = await engine.audioBuffer
                    XCTAssertEqual(chBefore, chAfter)
                    XCTAssertEqual(tmBefore, tmAfter)
                    XCTAssertEqual(lnBefore, lnAfter)
                    XCTAssertEqual(melBefore, melAfter)
                    XCTAssertEqual(hBefore, hAfter)
                    XCTAssertEqual(cBefore, cAfter)
                    XCTAssertEqual(tokBefore, tokAfter)
                    XCTAssertEqual(chunksBefore, chunksAfter)
                    XCTAssertEqual(bufBefore, bufAfter)

                    if let preview, !preview.tentativeText.isEmpty {
                        tentativeUpdates += 1
                        if firstTentativeSample == nil {
                            firstTentativeSample = preview.audioThroughSample
                        }
                    }
                }
            }

            let res = try await engine.finishWithRecognitionDetails()
            return (res, tentativeUpdates, firstTentativeSample, firstCommittedSample)
        }

        let baseline = try await runPass(withTentative: false)
        let candidate = try await runPass(withTentative: true)

        XCTAssertFalse(baseline.result.text.isEmpty)
        XCTAssertEqual(baseline.result.text, candidate.result.text)
        XCTAssertEqual(baseline.result.confidences, candidate.result.confidences)
        XCTAssertEqual(baseline.result.timestamps, candidate.result.timestamps)
        XCTAssertEqual(baseline.result.alternatives.count, candidate.result.alternatives.count)
        for (a, b) in zip(baseline.result.alternatives, candidate.result.alternatives) {
            XCTAssertEqual(a.map(\.tokenId), b.map(\.tokenId))
            XCTAssertEqual(a.map(\.probability), b.map(\.probability))
        }
        XCTAssertGreaterThan(candidate.tentativeUpdates, 0)
        if let tSample = candidate.firstTentativeSample, let cSample = baseline.firstCommittedSample {
            XCTAssertLessThan(tSample, cSample, "Tentative words should appear before the first committed chunk")
        }
    }
}

#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Transcribe a single WAV file with Nemotron, outputting top-N candidates per token.
public class NemotronTranscribe {
    private let logger = AppLogger(category: "NemotronTranscribe")

    public static func run(arguments: [String]) async {
        let logger = AppLogger(category: "NemotronTranscribe")

        guard let wavPath = arguments.first else {
            logger.error("Usage: nemotron-transcribe <wav-file> [--chunk-size 160] [--json]")
            return
        }

        let wavURL = URL(fileURLWithPath: wavPath)
        guard FileManager.default.fileExists(atPath: wavURL.path) else {
            logger.error("File not found: \(wavPath)")
            return
        }

        var chunkSize: NemotronChunkSize = .ms160
        var jsonOutput = false
        var modelDir: URL?

        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--chunk-size":
                i += 1
                if i < arguments.count, let ms = Int(arguments[i]),
                   let cs = NemotronChunkSize(rawValue: ms) {
                    chunkSize = cs
                }
            case "--model-dir":
                i += 1
                if i < arguments.count {
                    modelDir = URL(fileURLWithPath: arguments[i])
                }
            case "--json":
                jsonOutput = true
            default: break
            }
            i += 1
        }

        do {
            // Find or download models
            let manager = NemotronStreamingAsrManager()
            let resolvedModelDir: URL
            if let modelDir {
                resolvedModelDir = modelDir
            } else {
                let repo = chunkSize.repo
                let modelsBaseDir = FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent(".cache/fluidaudio/Models")
                let cacheDir = modelsBaseDir.appendingPathComponent(repo.folderName)
                if !FileManager.default.fileExists(atPath: cacheDir.path) {
                    logger.info("Downloading Nemotron \(chunkSize.rawValue)ms models...")
                    try await DownloadUtils.downloadRepo(repo, to: modelsBaseDir)
                }
                resolvedModelDir = cacheDir
            }

            logger.info("Loading Nemotron \(chunkSize.rawValue)ms from \(resolvedModelDir.path)...")
            try await manager.loadModels(modelDir: resolvedModelDir)
            logger.info("Models loaded")

            // Load audio
            let audioFile = try AVAudioFile(forReading: wavURL)
            let buffer = AVAudioPCMBuffer(
                pcmFormat: audioFile.processingFormat,
                frameCapacity: AVAudioFrameCount(audioFile.length)
            )!
            try audioFile.read(into: buffer)

            let audioDuration = Double(audioFile.length) / audioFile.processingFormat.sampleRate
            logger.info("Audio: \(String(format: "%.1f", audioDuration))s, \(Int(audioFile.processingFormat.sampleRate))Hz")

            // Process
            let startTime = Date()
            _ = try await manager.process(audioBuffer: buffer)
            let (transcript, confidences, alternatives) = try await manager.finish()
            let elapsed = Date().timeIntervalSince(startTime)

            logger.info("Transcribed in \(String(format: "%.0f", elapsed * 1000))ms")
            logger.info("Text: \(transcript)")

            // Pre-decode all token IDs to strings (both clean and raw with ▁ markers)
            var tokenStrings: [Int: String] = [:]
            var rawTokenStrings: [Int: String] = [:]
            for alts in alternatives {
                for alt in alts {
                    if tokenStrings[alt.tokenId] == nil {
                        tokenStrings[alt.tokenId] = await manager.decodeToken(alt.tokenId)
                        rawTokenStrings[alt.tokenId] = await manager.rawDecodeToken(alt.tokenId)
                    }
                }
            }
            let decode: (Int) -> String = { tokenStrings[$0] ?? "<\($0)>" }
            let rawDecode: (Int) -> String = { rawTokenStrings[$0] ?? "<\($0)>" }

            if jsonOutput {
                // Build JSON output with per-token details
                var tokens: [[String: Any]] = []
                for idx in 0..<alternatives.count {
                    let conf = idx < confidences.count ? confidences[idx] : 0
                    let alts = alternatives[idx]
                    let altDicts: [[String: Any]] = alts.map { alt in
                        [
                            "token": decode(alt.tokenId),
                            "raw_token": rawDecode(alt.tokenId),
                            "tokenId": alt.tokenId,
                            "probability": alt.probability,
                        ]
                    }
                    tokens.append([
                        "position": idx,
                        "confidence": conf,
                        "alternatives": altDicts,
                    ])
                }

                let output: [String: Any] = [
                    "file": wavURL.lastPathComponent,
                    "transcript": transcript,
                    "duration_s": audioDuration,
                    "inference_ms": Int(elapsed * 1000),
                    "token_count": confidences.count,
                    "tokens": tokens,
                ]

                if let jsonData = try? JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys]),
                   let jsonStr = String(data: jsonData, encoding: .utf8) {
                    print(jsonStr)
                }
            } else {
                // Human-readable output
                print("\nTranscript: \(transcript)")
                print("Duration:   \(String(format: "%.1f", audioDuration))s")
                print("Inference:  \(String(format: "%.0f", elapsed * 1000))ms")
                print("Tokens:     \(confidences.count)")
                print("")

                // Show tokens with low confidence and their alternatives
                for (idx, alts) in alternatives.enumerated() where alts.count > 1 {
                    let conf = idx < confidences.count ? confidences[idx] : 0
                    guard conf < 0.9 else { continue }  // only show uncertain tokens
                    let chosen = decode(alts[0].tokenId)
                    let altStr = alts.prefix(5).map { "\(decode($0.tokenId))(\(String(format: "%.2f", $0.probability)))" }.joined(separator: " ")
                    print("  [\(idx)] \(chosen) (conf=\(String(format: "%.3f", conf))): \(altStr)")
                }
            }
        } catch {
            logger.error("Failed: \(error)")
        }
    }
}
#endif

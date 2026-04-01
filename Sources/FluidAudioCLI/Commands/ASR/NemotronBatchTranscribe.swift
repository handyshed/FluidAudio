#if os(macOS)
import AVFoundation
import FluidAudio
import CoreML
import Foundation

/// Batch transcribe WAV files through Nemotron on ANE.
/// Outputs tab-separated: filename\ttranscript
public class NemotronBatchTranscribe {
    public static func run(arguments: [String]) async {
        let logger = AppLogger(category: "NemotronBatchTranscribe")

        guard arguments.count >= 1 else {
            logger.error("Usage: nemotron-batch <manifest.json> [--model-dir DIR] [--chunk-size 1120]")
            return
        }

        var manifestPath = arguments[0]
        var modelDir: URL?
        var chunkSize: NemotronChunkSize = .ms1120
        var withAlternatives = false

        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--model-dir":
                i += 1
                if i < arguments.count { modelDir = URL(fileURLWithPath: arguments[i]) }
            case "--chunk-size":
                i += 1
                if i < arguments.count, let ms = Int(arguments[i]),
                   let cs = NemotronChunkSize(rawValue: ms) { chunkSize = cs }
            case "--alternatives":
                withAlternatives = true
            default: break
            }
            i += 1
        }

        do {
            // Load manifest
            let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
            let manifest = try JSONSerialization.jsonObject(with: manifestData) as! [[String: Any]]
            logger.info("Loaded manifest: \(manifest.count) files")

            // Resolve model dir
            let resolvedModelDir: URL
            if let modelDir {
                resolvedModelDir = modelDir
            } else {
                let repo = chunkSize.repo
                let modelsBaseDir = FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent(".cache/fluidaudio/Models")
                let cacheDir = modelsBaseDir.appendingPathComponent(repo.folderName)
                if !FileManager.default.fileExists(atPath: cacheDir.path) {
                    logger.info("Downloading models...")
                    try await DownloadUtils.downloadRepo(repo, to: modelsBaseDir)
                }
                resolvedModelDir = cacheDir
            }

            // Load model with ANE enabled
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use ANE
            let manager = NemotronStreamingAsrManager(configuration: config)
            logger.info("Loading Nemotron \(chunkSize.rawValue)ms on ANE...")
            try await manager.loadModels(modelDir: resolvedModelDir)
            if withAlternatives {
                await manager.setSkipLogitSaving(false)
                logger.info("Models loaded on ANE (logit saving enabled for alternatives)")
            } else {
                await manager.setSkipLogitSaving(true)
                logger.info("Models loaded on ANE (logit saving disabled for speed)")
            }

            // Process files — write incrementally as JSONL (one JSON object per line)
            let outputPath = URL(fileURLWithPath: manifestPath)
                .deletingLastPathComponent()
                .appendingPathComponent("training_pairs.jsonl")

            // Clear output file
            FileManager.default.createFile(atPath: outputPath.path, contents: nil)
            let fileHandle = try FileHandle(forWritingTo: outputPath)
            defer { fileHandle.closeFile() }

            let startTime = Date()
            var successCount = 0
            var errorCount = 0

            for (idx, entry) in manifest.enumerated() {
                guard let text = entry["text"] as? String,
                      let filename = (entry["filename"] ?? entry["file"]) as? String else { continue }

                let audioPath = URL(fileURLWithPath: manifestPath)
                    .deletingLastPathComponent()
                    .appendingPathComponent("audio")
                    .appendingPathComponent(filename)

                guard FileManager.default.fileExists(atPath: audioPath.path) else {
                    logger.error("Missing: \(filename)")
                    errorCount += 1
                    continue
                }

                do {
                    let t0 = CFAbsoluteTimeGetCurrent()

                    // Read raw PCM directly — files are 16kHz mono 16-bit WAV
                    let fileData = try Data(contentsOf: audioPath)
                    let headerSize = 44  // Standard WAV header
                    guard fileData.count > headerSize else {
                        errorCount += 1
                        continue
                    }
                    let pcmData = fileData.subdata(in: headerSize..<fileData.count)

                    // Convert Int16 → Float32
                    let int16Count = pcmData.count / 2
                    let floatSamples: [Float] = pcmData.withUnsafeBytes { raw in
                        let int16Ptr = raw.bindMemory(to: Int16.self)
                        return (0..<int16Count).map { Float(int16Ptr[$0]) / 32768.0 }
                    }
                    let ioMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

                    let t1 = CFAbsoluteTimeGetCurrent()
                    _ = try await manager.processSamples(floatSamples)
                    let processMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000

                    let t2 = CFAbsoluteTimeGetCurrent()
                    let (transcript, confidences, alternatives, _) = try await manager.finish()
                    let finishMs = (CFAbsoluteTimeGetCurrent() - t2) * 1000

                    let t3 = CFAbsoluteTimeGetCurrent()
                    await manager.resetFast()
                    let resetMs = (CFAbsoluteTimeGetCurrent() - t3) * 1000

                    if idx < 5 || idx % 200 == 0 {
                        logger.info("  Timing: io=\(String(format: "%.0f", ioMs))ms process=\(String(format: "%.0f", processMs))ms finish=\(String(format: "%.0f", finishMs))ms reset=\(String(format: "%.0f", resetMs))ms")
                    }

                    // Build output
                    var pair: [String: Any] = ["clean": text, "noisy": transcript, "file": filename]

                    if withAlternatives {
                        // Decode token strings
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

                        var tokens: [[String: Any]] = []
                        for (tidx, alts) in alternatives.enumerated() {
                            let conf = tidx < confidences.count ? confidences[tidx] : Float(0)
                            let altDicts: [[String: Any]] = alts.map { alt in
                                [
                                    "t": tokenStrings[alt.tokenId] ?? "<\(alt.tokenId)>",
                                    "r": rawTokenStrings[alt.tokenId] ?? "",
                                    "p": alt.probability,
                                ]
                            }
                            tokens.append(["c": conf, "a": altDicts])
                        }
                        pair["tokens"] = tokens
                    }

                    if let jsonData = try? JSONSerialization.data(withJSONObject: pair),
                       let jsonStr = String(data: jsonData, encoding: .utf8) {
                        fileHandle.write((jsonStr + "\n").data(using: .utf8)!)
                    }
                    successCount += 1

                    if (idx + 1) % 50 == 0 || idx == 0 {
                        let elapsed = Date().timeIntervalSince(startTime)
                        let rate = Double(idx + 1) / elapsed
                        let eta = Double(manifest.count - idx - 1) / rate
                        logger.info("\(idx + 1)/\(manifest.count) (\(String(format: "%.1f", rate))/s, ETA \(String(format: "%.0f", eta))s) — \(successCount) ok, \(errorCount) err")
                    }
                } catch {
                    logger.error("Failed \(filename): \(error)")
                    errorCount += 1
                    await manager.reset()
                }
            }

            let elapsed = Date().timeIntervalSince(startTime)
            logger.info("Done: \(successCount) pairs in \(String(format: "%.0f", elapsed))s → \(outputPath.path)")

        } catch {
            logger.error("Failed: \(error)")
        }
    }
}
#endif

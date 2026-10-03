@preconcurrency import CoreML
import XCTest
@testable import FluidAudio

final class NemotronRecognitionLogitsTests: XCTestCase {
    func testMaximumIndexSupportsFloat16AndFloat32Logits() async throws {
        let manager = StreamingNemotronAsrManager()
        for type in [MLMultiArrayDataType.float16, .float32] {
            let logits = try MLMultiArray(shape: [4], dataType: type)
            for (index, value) in [-10, 2, 9, -3].enumerated() {
                logits[index] = NSNumber(value: value)
            }
            let maximum = await manager.findMaxIndex(logits)
            XCTAssertEqual(maximum, 2)
        }
    }

    func testLogicalLogitsExcludeStridedPadding() throws {
        let storage = UnsafeMutablePointer<Float>.allocate(capacity: 8)
        storage.initialize(repeating: 10000, count: 8)
        storage[0] = -10
        storage[2] = 2
        storage[4] = 9
        storage[6] = -3
        let logits = try MLMultiArray(dataPointer: storage, shape: [4], dataType: .float32, strides: [2]) {
            $0.assumingMemoryBound(to: Float.self).deallocate()
        }
        let values = NemotronRecognitionLogits.values(from: logits)
        XCTAssertEqual(values, [-10, 2, 9, -3])
        XCTAssertGreaterThan(NemotronRecognitionLogits.confidence(values, token: 2), 0.99)
    }

    func testConfidenceRemainsFiniteForLargeLogits() {
        let values: [Float] = [10000, 9999, -10000]
        let confidence = NemotronRecognitionLogits.confidence(values, token: 0)
        XCTAssertTrue(confidence.isFinite)
        XCTAssertEqual(confidence, 1 / (1 + exp(Float(-1))), accuracy: 0.000001)
    }

    func testAlternativesExcludeBlankWithoutRenormalizing() {
        let values: [Float] = [0, 0, 0, 0]
        let candidates = NemotronRecognitionLogits.alternatives(values, blank: 3)
        XCTAssertEqual(candidates.map(\.tokenId), [0, 1, 2])
        XCTAssertEqual(candidates.reduce(Float(0)) { $0 + $1.probability }, 0.75)
        XCTAssertTrue(candidates.allSatisfy { $0.probability == 0.25 })
    }

    func testTop16IncludesBlankInRankingBeforeFiltering() {
        let values = (0..<20).map(Float.init)
        let candidates = NemotronRecognitionLogits.alternatives(values, blank: 19)
        XCTAssertEqual(candidates.count, 15)
        XCTAssertEqual(candidates.first?.tokenId, 18)
        XCTAssertEqual(candidates.last?.tokenId, 4)
        XCTAssertEqual(
            candidates[0].probability,
            NemotronRecognitionLogits.confidence(values, token: 18), accuracy: 0.000001)
    }

    func testEmptyInputAndInvalidTokenDoNotProduceNaN() {
        XCTAssertEqual(NemotronRecognitionLogits.confidence([], token: 0), 0)
        XCTAssertEqual(NemotronRecognitionLogits.confidence([1], token: 1), 0)
        XCTAssertTrue(NemotronRecognitionLogits.alternatives([], blank: 0).isEmpty)
        XCTAssertTrue(NemotronRecognitionLogits.alternatives([1], blank: 0, count: 0).isEmpty)
    }
}

@preconcurrency import CoreML
import Foundation

/// A candidate from the recognition network's vocabulary distribution.
public struct TokenCandidate: Sendable, Equatable {
    public let tokenId: Int
    public let probability: Float
    public init(tokenId: Int, probability: Float) {
        self.tokenId = tokenId
        self.probability = probability
    }
}

/// An immutable view of a streaming recognition session.
public struct StreamingTranscriptSnapshot: Sendable {
    public let revision: Int
    public let text: String
    public let tokenIds: [Int]
    public let confidences: [Float]
    /// Absolute token emission times in milliseconds.
    public let timestampsMs: [Int]
    public let isFinal: Bool
}
public typealias NemotronSnapshotCallback = @Sendable (StreamingTranscriptSnapshot) -> Void

/// Final recognition text and aligned per-token correction signals.
public struct NemotronRecognitionDetails: Sendable {
    public let text: String
    public let tokenIds: [Int]
    public let confidences: [Float]
    public let alternatives: [[TokenCandidate]]
    public let timestamps: [Int]
}

/// Stable softmax extraction shared by streamed and final tokens.
internal enum NemotronRecognitionLogits {
    static func values(from logits: MLMultiArray) -> [Float] {
        // CoreML buffers may include padding. Read logical elements rather than allocation contents.
        (0..<logits.count).map { logits[$0].floatValue }
    }

    static func confidence(_ values: [Float], token: Int) -> Float {
        guard let maximum = values.max(), values.indices.contains(token) else { return 0 }
        let denominator = values.reduce(Float(0)) { $0 + exp($1 - maximum) }
        return exp(values[token] - maximum) / denominator
    }

    static func alternatives(_ values: [Float], blank: Int, count: Int = 16) -> [TokenCandidate] {
        guard let maximum = values.max(), count > 0 else { return [] }
        let denominator = values.reduce(Float(0)) { $0 + exp($1 - maximum) }
        // Preserve the fork's top-K contract: rank the full distribution, then remove blank.
        var candidates: [TokenCandidate] = []
        candidates.reserveCapacity(values.count)
        for (token, logit) in values.enumerated() {
            let probability: Float = exp(logit - maximum) / denominator
            candidates.append(TokenCandidate(tokenId: token, probability: probability))
        }
        candidates.sort { left, right in
            if left.probability == right.probability {
                return left.tokenId < right.tokenId
            }
            return left.probability > right.probability
        }
        let topCandidates = candidates.prefix(count)
        return topCandidates.filter { $0.tokenId != blank }
    }
}

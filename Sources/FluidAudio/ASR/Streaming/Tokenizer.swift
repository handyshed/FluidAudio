import Foundation

public class Tokenizer {
    private var vocab: [String: String] = [:]
    private var idToToken: [Int: String] = [:]

    public init(vocabPath: URL) throws {
        let data = try Data(contentsOf: vocabPath)
        let json = try JSONSerialization.jsonObject(with: data, options: []) as! [String: String]

        self.vocab = json
        for (key, value) in json {
            if let id = Int(key) {
                self.idToToken[id] = value
            }
        }
    }

    public func decode(ids: [Int]) -> String {
        var text = ""
        for id in ids {
            if let token = idToToken[id] {
                text += token
            }
        }
        // Replace SentencePiece word boundary marker with space, then trim
        var result = text.replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)

        // Clean up sentence boundary artifacts from multi-sentence decoding:
        // Remove stray tokens between sentence-end punctuation and next word
        // e.g. "fine.- but" → "fine. But", "this?ice three" → "this? Three"
        let pattern = "([.?!])([^\\s.?!A-Z][^\\s]*)?\\s+"
        if let regex = try? NSRegularExpression(pattern: pattern) {
            result = regex.stringByReplacingMatches(
                in: result, range: NSRange(result.startIndex..., in: result),
                withTemplate: "$1 "
            )
        }

        // Capitalize first letter after sentence-ending punctuation
        let capPattern = "([.?!])\\s+([a-z])"
        if let regex = try? NSRegularExpression(pattern: capPattern) {
            let mutable = NSMutableString(string: result)
            let matches = regex.matches(in: result, range: NSRange(result.startIndex..., in: result))
            for match in matches.reversed() {
                let letterRange = match.range(at: 2)
                if let range = Range(letterRange, in: result) {
                    let upper = result[range].uppercased()
                    mutable.replaceCharacters(in: letterRange, with: upper)
                }
            }
            result = mutable as String
        }

        return result
    }

    /// Decode a single token ID preserving the raw SentencePiece representation.
    /// The ▁ prefix indicates a word boundary (start of a new word).
    public func rawToken(id: Int) -> String {
        idToToken[id] ?? "<\(id)>"
    }
}

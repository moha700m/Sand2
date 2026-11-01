//
// FastVLM.swift
// IOSLocalLLM
//
// Adapted from apple/ml-fastvlm — app/FastVLM/FastVLM.swift
// © 2025 Apple Inc. All Rights Reserved. (original)
// Adaptations for IOSLocalLLM:
//   - MediaProcessingExtensions → MediaProcessing (MLXVLM 2.29.1 public API)
//   - LMInput.ProcessedImage(imageGridThw:) → (frames:)  (API change in 2.29.1)
//   - Added Prompt.asMessages() extension (not in 2.29.1 public API)
//

import CoreImage
import CoreML
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM
// NOTE: `Tokenizer` now comes from MLXLMCommon — mlx-swift-lm 3.x defines its
// own `Tokenizer` protocol, so also importing swift-transformers' `Tokenizers`
// made the type ambiguous. The processor registry hands us an
// `MLXLMCommon.Tokenizer`; `.encode(text:)` is provided there.

// MARK: - UserInput.Prompt → messages helper (not in MLXVLM 2.29.1 public API)

private extension UserInput.Prompt {
    /// Converts any Prompt variant to a flat [role: content] array for the FastVLM chat template.
    func asMessages() -> [[String: String]] {
        switch self {
        case .text(let text):
            return [["role": "user", "content": text]]
        case .messages(let msgs):
            return msgs.compactMap { msg in
                guard let role = msg["role"] as? String else { return nil }
                let content = (msg["content"] as? String) ?? ""
                return ["role": role, "content": content]
            }
        case .chat(let chatMessages):
            return chatMessages.map { ["role": $0.role.rawValue, "content": $0.content] }
        }
    }
}

// MARK: - rotateHalf (RoPE helper)

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let index = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0 ..< index]
    let x2 = x[.ellipsis, index...]
    return concatenated([-x2, x1], axis: -1)
}

// MARK: - Language (Qwen2 decoder in MLX)

private enum Language {

    static private func applyMultimodalRotaryPositionEmb
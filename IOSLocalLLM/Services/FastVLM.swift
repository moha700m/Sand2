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
    /// Converts any Prompt variant to a flat [role: cont
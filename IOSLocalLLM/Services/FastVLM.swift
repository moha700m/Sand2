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

    static private func applyMultimodalRotaryPositionEmbedding(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray,
        positionIds: MLXArray, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        var cos = cos[positionIds]
        var sin = sin[positionIds]

        cos = concatenated(
            split(cos, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
            axis: -1
        )[0..., .newAxis, 0..., 0...]

        sin = concatenated(
            split(sin, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
            axis: -1
        )[0..., .newAxis, 0..., 0...]

        let qEmbed = (q * cos) + (rotateHalf(q) * sin)
        let kEmbed = (k * cos) + (rotateHalf(k) * sin)
        return (qEmbed, kEmbed)
    }

    fileprivate class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let mropeSection: [Int]

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ args: FastVLMConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            if let v = args.ropeScaling?["mrope_section"], let array = v.asInts() {
                self.mropeSection = sequence(state: (0, array.makeIterator())) { state in
                    if let v = state.1.next() {
                        state.0 += v * 2
                        return state.0
                    } else {
                        return nil
                    }
                }.dropLast()
            } else {
                // Fallback: default mrope_section for Qwen2-0.5B [2,1,1] → cumsum*2 = [4,6]
                self.mropeSection = [4, 6]
            }

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: (any KVCache)?) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            let mask = mask?[0..., 0 ..< keys.dim(-2)]

            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

            if let cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }

            let attnOut = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return wo(attnOut)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class DecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ args: FastVLMConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: (any KVCache)?) -> MLXArray {
            let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    fileprivate class Qwen2Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        fileprivate let layers: [DecoderLayer]
        fileprivate let norm: RMSNorm

        init(_ args: FastVLMConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            self.layers = (0 ..< args.hiddenLayers).map { _ in DecoderLayer(args) }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        func callAsFunction(
        
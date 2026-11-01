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
            _ inputs: MLXArray?, cache: [any KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            // Explicitly request MLXArray? form (there's a ScaledDotProductAttentionMaskMode
            // overload in MLXLMCommon that would otherwise be ambiguous).
            let mask: MLXArray? = createAttentionMask(h: h, cache: cache)
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }
            return norm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {

        @ModuleInfo var model: Qwen2Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear
        var kvHeads: [Int]

        init(_ args: FastVLMConfiguration.TextConfiguration) {
            self.model = Qwen2Model(args)
            // Always allocate lm_head. Different FastVLM repos disagree about
            // `tie_word_embeddings` — Apple's checkpoints ship lm_head.weight
            // explicitly while some MLX repacks declare tied embeddings yet
            // still emit the tensor anyway. Keeping the module present means
            // sanitize can either load the shipped weight or alias
            // embed_tokens into it, instead of crashing the load with
            // "language_model.lm_head: none not compatible with [weight: …]".
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [any KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            let h = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            return LMOutput(logits: lmHead(h))
        }
    }
}

// MARK: - Vision (Core ML encoder bridge)

private enum Vision {

    /// Wraps the auto-generated `fastvithd` Core ML class.
    /// The mlpackage is compiled into the app bundle by Xcode.
    fileprivate class VisionModelCoreML {

        let lock = NSLock()
        var _model: fastvithd?

        func load() throws -> fastvithd {
            try lock.withLock {
                if let model = _model { return model }
                let model = try fastvithd()
                _model = model
                return model
            }
        }

        /// Encodes a [1, 3, H, W] float32 MLXArray through the Core ML vision encoder.
        /// Returns [1, 256, 3072] float32.
        ///
        /// On ANY Core ML failure — the model failed to load/compile, OOM, a
        /// prediction error — this returns a zero-filled feature map instead of
        /// crashing. This path runs on the live Lens loop via the MLX module
        /// graph (a non-throwing `callAsFunction`), so a thrown error here had
        /// nowhere to go but `try!` → SIGABRT, taking the whole app down the
        /// instant the camera saw a frame with a bad/half-loaded encoder. A
        /// poor caption beats a hard crash; the failure is recorded so it's
        /// diagnosable rather than silent.
        func encode(_ image: MLXArray) -> MLXArray {
            func zeroFeatures() -> MLXArray {
                let buf = [Float32](repeating: 0, count: 1 * 256 * 3072)
                return buf.withUnsafeBytes { MLXArray($0, [1, 256, 3072], type: Float32.self) }
            }

            guard let coreMLModel = try? load() else {
                Diagnostics.shared.breadcrumb(
                    "FastVLM vision encoder unavailable — Core ML model failed to load", category: "lens")
                return zeroFeatures()
            }

            precondition(image.ndim == 4)
            precondition(image.dim(0) == 1)
            precondition(image.dim(1) == 3)

            let H = image.dim(2)
            let W = image.dim(3)
            let expectedCount = 1 * 3 * H * W

            // Force a contiguous float32 host copy. Zero-copy into MLX-backed
            // storage + default Core ML `.all` compute units previously routed
            // through MetalPerformanceShadersGraph and SIGSEGV'd mid-prediction
            // on iOS 27. Owning the buffer + ANE/CPU compute units (see
            // `fastvithd.init`) removes both crash vectors.
            let contiguous = image.asType(.float32).asData(access: .copy)
            let shape: [NSNumber] = [1, 3, NSNumber(value: H), NSNumber(value: W)]
            guard let owned = try? MLMultiArray(shape: shape, dataType: .float32) else {
                Diagnostics.shared.breadcrumb(
                    "FastVLM encode failed: could not allocate MLMultiArray", category: "lens")
                return zeroFeatures()
            }
            contiguous.data.withUnsafeBytes { src in
                guard let base = src.baseAddress else { return }
                memcpy(owned.dataPointer, base,
                       min(src.count, expectedCount * MemoryLayout<Float32>.size))
            }

            do {
                let output = try coreMLModel.prediction(images: owned)
                precondition(output.image_features.shape == [1, 256, 3072])
                return output.image_features.withUnsafeBytes { ptr in
                    MLXArray(ptr, [1, 256, 3072], type: Float32.self)
                }
            } catch {
                Diagnostics.shared.breadcrumb(
                    "FastVLM encode failed: \(error.localizedDescription)", category: "lens")
                return zeroFeatures()
            }
        }
    }

    fileprivate class VisionModel: Module {

        let model = VisionModelCoreML()

        override init() {}

        func callAsFunction(_ hiddenStates: MLXArray, gridThw: [THW]) -> MLXArray {
            model.encode(hiddenStates)
        }
    }
}

// MARK: - FastVLMProcessor

/// `UserInputProcessor` for FastVLM / LlavaQwen2.
///
/// Registered as "LlavaProcessor" by ``FastVLM/register(modelFactory:)``.
public final class FastVLMProcessor: UserInputProcessor {

    private let config: FastVLMProcessorConfiguration
    private let imageProcessingConfig: FastVLMPreProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: FastVLMPreProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = FastVLMProcessorConfiguration()
        self.imageProcessingConfig = config
        self.tokenizer = tokenizer
    }

    /// Pre-processes a CIImage: resize to shortest edge, center-crop, normalize.
    /// Returns the [1, C, H, W] pixel array and its THW descriptor.
    public func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (MLXArray, THW) {
        var img = MediaProcessing.apply(im
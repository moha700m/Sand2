# Sand2

**On-device local LLM for iPhone** — chat, vision, voice, RAG, and an
authenticated local API. SwiftUI workbench with MLX, llama.cpp, whisper.cpp,
and Core ML.

> Parity port of [Mesutcydev/ios-local-llm](https://github.com/Mesutcydev/ios-local-llm) (MIT).
> See [ATTRIBUTION.md](ATTRIBUTION.md).

## Build

```bash
git clone --recurse-submodules https://github.com/moha700m/Sand2.git
cd Sand2
brew install xcodegen cocoapods cmake
./scripts/build_native_frameworks.sh
xcodegen generate
pod install
open OnDeviceLAS.xcworkspace
```

Select `OnDeviceLAS` or `IOSLocalLLM`. Change bundle ID/team for devices.

## License

MIT — see `LICENSE` and `ATTRIBUTION.md`.

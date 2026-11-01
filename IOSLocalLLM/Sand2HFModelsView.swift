import SwiftUI

/// Simple Hugging Face browser: curated starters + live search + install.
struct Sand2HFModelsView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var search = HFSearchService()
    @ObservedObject private var downloadCenter = ModelDownloadCenter.shared
    @ObservedObject private var modelService = CodingAssistantService.shared

    @State private var query = "mlx-community qwen"
    @State private var didInitialSearch = false

    private let starters: [(title: String, repo: String, note: String)] = [
        ("Qwen3 1.7B", "mlx-community/Qwen3-1.7B-4bit", "خفيف وسريع"),
        ("Qwen3 4B", "mlx-community/Qwen3-4B-Instruct-2507-4bit", "موصى به"),
        ("Qwen2.5 Coder 1.5B", "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit", "للبرمجة"),
        ("Qwen3 0.6B", "mlx-community/Qwen3-0.6B-4bit", "الأصغر")
    ]

    var body: some View {
        NavigationStack {
            ZStack {
                Sand2PageBackground()

                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        Sand2SectionLabel(text: "موصى بها للجهاز")
                        ForEach(starters, id: \.repo) { item in
                            starterRow(item)
                        }

                        Sand2SectionLabel(text: "بحث Hugging Face")
                        searchField

                        if search.isSearching {
                            ProgressView("جارٍ البحث…")
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 20)
                        } else if let err = search.lastError, !err.isEmpty {
                            Text(err)
                                .font(.system(.footnote, design: .rounded))
                                .foregroundStyle(Sand2Theme.danger)
                        } else {
                            ForEach(search.results) { model in
                                resultRow(model)
                            }
                        }

                        if !installed.isEmpty {
                            Sand2SectionLabel(text: "مثبت على الجهاز")
                            ForEach(installed, id: \.id) { model in
                                installedRow(model)
                            }
                        }
                    }
                    .padding(Sand2Theme.pagePad)
                }
            }
            .navigationTitle("موديلات")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("إغلاق") { dismiss() }
                        .font(.system(.body, design: .rounded).weight(.semibold))
                }
            }
            .task {
                if !didInitialSearch {
                    didInitialSearch = true
                    await search.search(query: query, filter: .mlx, limit: 20)
                }
            }
        }
    }

    private var installed: [DownloadableModel] {
        downloadCenter.models.filter { $0.isReady }
    }

    private var searchField: some View {
        HStack(spacing: 10) {
            TextField("مثال: mlx-community/Qwen3", text: $query)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .font(.system(.body, design: .rounded))
                .padding(.horizontal, 12)
                .frame(minHeight: 46)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(Sand2Theme.surface)
                        .overlay(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .stroke(Sand2Theme.stroke, lineWidth: 1)
                        )
                )

            Button {
                Task { await search.search(query: query, filter: .mlx, limit: 25) }
            } label: {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.white)
                    .frame(width: 46, height: 46)
                    .background(Sand2Theme.accent, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            .buttonStyle(.plain)
        }
    }

    private func starterRow(_ item: (title: String, repo: String, note: String)) -> some View {
        Sand2HFDownloadRow(
            title: item.title,
            subtitle: item.repo,
            badge: item.note,
            summary: HFModelSummary.preview(id: item.repo)
        )
    }

    private func resultRow(_ model: HFModelSummary) -> some View {
        Sand2HFDownloadRow(
            title: model.modelName,
            subtitle: model.id,
            badge: model.pipelineTag ?? "MLX",
            summary: model
        )
    }

    private func installedRow(_ model: DownloadableModel) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(model.displayName)
                    .font(.system(.subheadline, design: .rounded).weight(.semibold))
                    .foregroundStyle(Sand2Theme.ink)
                Text(model.sourceRepoID)
                    .font(.system(.caption2, design: .rounded))
                    .foregroundStyle(Sand2Theme.inkMuted)
                    .lineLimit(1)
            }
            Spacer()
            Button("تفعيل") {
                activateInstalled(model)
            }
            .font(.system(.caption, design: .rounded).weight(.bold))
            .foregroundStyle(.white)
            .padding(.horizontal, 12)
            .frame(minHeight: 34)
            .background(Sand2Theme.accent, in: Capsule())
            .buttonStyle(.plain)
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: Sand2Theme.radius, style: .continuous)
                .fill(Sand2Theme.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Sand2Theme.radius, style: .continuous)
                        .stroke(Sand2Theme.stroke, lineWidth: 1)
                )
        )
    }

    private func activateInstalled(_ model: DownloadableModel) {
        let stored = "downloaded:\(model.sourceRepoID)"
        if let assistant = AssistantModelCatalog.selection(forStoredID: stored)
            ?? AssistantModelCatalog.selection(forStoredID: "custom:\(model.sourceRepoID)")
            ?? AssistantModelCatalog.presets.first(where: { $0.repoID == model.sourceRepoID }) {
            Task {
                await modelService.switchTo(assistant, persistAsDefault: true)
                dismiss()
            }
            return
        }

        if let preset = AssistantModelCatalog.presets.first(where: {
            $0.repoID.caseInsensitiveCompare(model.sourceRepoID) == .orderedSame
        }) {
            Task {
                await modelService.switchTo(preset, persistAsDefault: true)
                dismiss()
            }
        } else {
            AppSettings.shared.assistantModelID = stored
            AppSettings.shared.hasPickedAssistantModel = true
            modelService.startLoad()
            dismiss()
        }
    }
}

private struct Sand2HFDownloadRow: View {
    let title: String
    let subtitle: String
    let badge: String
    let summary: HFModelSummary

    @StateObject private var downloader: HFModelDownloadManager

    init(title: String, subtitle: String, badge: String, summary: HFModelSummary) {
        self.title = title
        self.subtitle = subtitle
        self.badge = badge
        self.summary = summary
        if let existing = ModelDownloadCenter.shared.existingDownloader(forRepoID: summary.id) {
            _downloader = StateObject(wrappedValue: existing)
        } else {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let dest = docs
                .appendingPathComponent("HFModels", isDirectory: true)
                .appendingPathComponent(summary.id.replacingOccurrences(of: "/", with: "_"))
            _downloader = StateObject(
                wrappedValue: HFModelDownloadManager(repoID: summary.id, destination: dest)
            )
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.system(.subheadline, design: .rounded).weight(.semibold))
                        .foregroundStyle(Sand2Theme.ink)
                    Text(subtitle)
                        .font(.system(.caption2, design: .rounded))
                        .foregroundStyle(Sand2Theme.inkMuted)
                        .lineLimit(2)
                }
                Spacer()
                Text(badge)
                    .font(.system(.caption2, design: .rounded).weight(.bold))
                    .foregroundStyle(Sand2Theme.accent)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Sand2Theme.accentSoft, in: Capsule())
            }

            progressOrButton
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: Sand2Theme.radius, style: .continuous)
                .fill(Sand2Theme.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: Sand2Theme.radius, style: .continuous)
                        .stroke(Sand2Theme.stroke, lineWidth: 1)
                )
        )
    }

    @ViewBuilder
    private var progressOrButton: some View {
        switch downloader.state {
        case .ready:
            Label("تم التنزيل", systemImage: "checkmark.circle.fill")
                .font(.system(.caption, design: .rounded).weight(.semibold))
                .foregroundStyle(Sand2Theme.good)
        case .downloading, .paused:
            VStack(alignment: .leading, spacing: 6) {
                ProgressView(value: downloader.progress)
                    .tint(Sand2Theme.accent)
                HStack {
                    Text("\(Int(downloader.progress * 100))٪")
                        .font(.system(.caption2, design: .rounded).weight(.bold))
                        .foregroundStyle(Sand2Theme.inkMuted)
                    Spacer()
                    Button("إلغاء") { downloader.cancel() }
                        .font(.system(.caption2, design: .rounded).weight(.semibold))
                }
            }
        case .failed:
            Button("إعادة المحاولة") { startDownload() }
                .font(.system(.caption, design: .rounded).weight(.bold))
                .foregroundStyle(Sand2Theme.danger)
        default:
            Button(action: startDownload) {
                Text("تنزيل من Hugging Face")
                    .font(.system(.caption, design: .rounded).weight(.bold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, minHeight: 40)
                    .background(Sand2Theme.accent, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            .buttonStyle(.plain)
        }
    }

    private func startDownload() {
        ModelDownloadCenter.shared.registerCustom(
            repoID: summary.id,
            displayName: title,
            subtitle: summary.id,
            category: .assistant,
            sizeLabel: "HF",
            docURL: "https://huggingface.co/\(summary.id)",
            downloader: downloader,
            runtime: .mlx
        )
        downloader.start()
    }
}

private extension HFModelSummary {
    static func preview(id: String) -> HFModelSummary {
        let parts = id.split(separator: "/", maxSplits: 1).map(String.init)
        let payload: [String: Any] = [
            "modelId": id,
            "id": id,
            "downloads": 0,
            "likes": 0,
            "pipeline_tag": "text-generation",
            "tags": ["mlx"],
            "author": parts.first ?? ""
        ]
        return HFModelSummary(json: payload)
    }
}

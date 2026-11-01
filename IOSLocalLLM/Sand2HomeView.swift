import SwiftUI

/// Arabic-first home: clear welcome chat CTA, HF models, simple conversation.
struct Sand2HomeView: View {
    @ObservedObject private var modelService = CodingAssistantService.shared
    @ObservedObject private var downloadCenter = ModelDownloadCenter.shared
    @State private var draft = ""
    @State private var messages: [ChatMessage] = []
    @State private var showHF = false
    @State private var isSending = false
    @State private var didGreet = false
    @FocusState private var focused: Bool

    var body: some View {
        NavigationStack {
            ZStack {
                Sand2PageBackground()

                VStack(spacing: 0) {
                    header
                    modelStrip
                    Divider().opacity(0.15)

                    if messages.isEmpty {
                        welcomePane
                    } else {
                        chatArea
                    }

                    composer
                }
            }
            .navigationBarHidden(true)
            .sheet(isPresented: $showHF) {
                Sand2HFModelsView()
                    .environment(\.layoutDirection, .rightToLeft)
                    .environment(\.locale, Locale(identifier: "ar_SA"))
            }
            .task {
                LocalizationService.shared.setLanguage(.arabic)
                downloadCenter.reconcileInstalledRegistry()
            }
        }
        .environment(\.layoutDirection, .rightToLeft)
        .environment(\.locale, Locale(identifier: "ar_SA"))
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Sand2")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundStyle(Sand2Theme.ink)
                Text("محلي · عربي · من Hugging Face")
                    .font(.system(.caption, design: .rounded).weight(.semibold))
                    .foregroundStyle(Sand2Theme.inkMuted)
            }
            Spacer()
            Sand2SecondaryButton(title: "موديلات", systemImage: "arrow.down.circle") {
                showHF = true
            }
        }
        .padding(.horizontal, Sand2Theme.pagePad)
        .padding(.top, 10)
        .padding(.bottom, 8)
    }

    private var modelStrip: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(statusColor)
                .frame(width: 10, height: 10)

            VStack(alignment: .leading, spacing: 2) {
                Text(modelService.activeDisplayName)
                    .font(.system(.subheadline, design: .rounded).weight(.semibold))
                    .foregroundStyle(Sand2Theme.ink)
                    .lineLimit(1)
                Text(statusText)
                    .font(.system(.caption2, design: .rounded))
                    .foregroundStyle(Sand2Theme.inkMuted)
            }

            Spacer()

            Button(action: toggleLoad) {
                Text(loadButtonTitle)
                    .font(.system(.caption, design: .rounded).weight(.bold))
                    .foregroundStyle(Sand2Theme.accent)
                    .padding(.horizontal, 12)
                    .frame(minHeight: 34)
                    .background(Sand2Theme.accentSoft, in: Capsule())
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, Sand2Theme.pagePad)
        .padding(.vertical, 10)
    }

    /// Clear, welcoming conversation entry — the primary action on first open.
    private var welcomePane: some View {
        VStack(alignment: .leading, spacing: 20) {
            Spacer(minLength: 28)

            Text("هلا فيك")
                .font(.system(size: 34, weight: .bold, design: .rounded))
                .foregroundStyle(Sand2Theme.ink)

            Text("أنا Sand2 — مساعد ذكي يشتغل على جهازك. حمّل موديل من Hugging Face، بعدين اضغط زر المحادثة وابدأ.")
                .font(.system(.body, design: .rounded))
                .foregroundStyle(Sand2Theme.inkMuted)
                .lineSpacing(4)

            Sand2PrimaryButton(
                title: "ابدأ محادثة",
                systemImage: "bubble.left.and.bubble.right.fill"
            ) {
                startWelcomingChat()
            }
            .accessibilityLabel("ابدأ محادثة")
            .accessibilityHint("يفتح محادثة جديدة مع رسالة ترحيب")

            Sand2SecondaryButton(title: "تصفّح موديلات Hugging Face", systemImage: "magnifyingglass") {
                showHF = true
            }

            Spacer()
        }
        .padding(.horizontal, Sand2Theme.pagePad)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var chatArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(messages) { message in
                        bubble(message)
                            .id(message.id)
                    }
                }
                .padding(.horizontal, Sand2Theme.pagePad)
                .padding(.vertical, 16)
            }
            .onChange(of: messages.last?.content) { _, _ in
                if let id = messages.last?.id {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private func bubble(_ message: ChatMessage) -> some View {
        let isUser = message.role == .user
        return HStack {
            if isUser { Spacer(minLength: 36) }
            Text(message.content.isEmpty && message.isStreaming ? "…" : message.content)
                .font(.system(.body, design: .rounded))
                .foregroundStyle(isUser ? Color.white : Sand2Theme.ink)
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(isUser ? Sand2Theme.accent : Sand2Theme.surface)
                        .overlay(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .stroke(isUser ? Color.clear : Sand2Theme.stroke, lineWidth: 1)
                        )
                )
            if !isUser { Spacer(minLength: 36) }
        }
    }

    private var composer: some View {
        VStack(spacing: 8) {
            if !messages.isEmpty {
                HStack {
                    Button {
                        startWelcomingChat()
                    } label: {
                        Label("محادثة جديدة", systemImage: "plus.bubble.fill")
                            .font(.system(.caption, design: .rounded).weight(.bold))
                            .foregroundStyle(Sand2Theme.accent)
                    }
                    .buttonStyle(.plain)
                    Spacer()
                }
            }

            HStack(alignment: .bottom, spacing: 10) {
                TextField("اكتب سؤالك…", text: $draft, axis: .vertical)
                    .font(.system(.body, design: .rounded))
                    .lineLimit(1...5)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(Sand2Theme.surface)
                            .overlay(
                                RoundedRectangle(cornerRadius: 16, style: .continuous)
                                    .stroke(Sand2Theme.stroke, lineWidth: 1)
                            )
                    )
                    .focused($focused)

                Button(action: send) {
                    Image(systemName: isSending ? "stop.fill" : "arrow.up")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 46, height: 46)
                        .background(Sand2Theme.accent, in: Circle())
                }
                .buttonStyle(.plain)
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isSending)
                .accessibilityLabel(isSending ? "إيقاف" : "إرسال")
            }
        }
        .padding(.horizontal, Sand2Theme.pagePad)
        .padding(.top, 8)
        .padding(.bottom, 12)
        .background(.ultraThinMaterial)
    }

    private var statusColor: Color {
        switch modelService.state {
        case .ready: return Sand2Theme.good
        case .loading, .generating: return Color.orange
        case .failed: return Sand2Theme.danger
        case .unloaded: return Sand2Theme.inkMuted
        }
    }

    private var statusText: String {
        switch modelService.state {
        case .ready: return "جاهز للرد"
        case .loading: return "جارٍ التحميل إلى الذاكرة…"
        case .generating: return "يكتب…"
        case .failed(let msg): return "فشل: \(msg)"
        case .unloaded: return "غير محمّل — اضغط تحميل أو ابدأ محادثة"
        }
    }

    private var loadButtonTitle: String {
        switch modelService.state {
        case .ready, .generating: return "إلغاء التحميل"
        case .loading: return "انتظر"
        case .unloaded, .failed: return "تحميل"
        }
    }

    private func toggleLoad() {
        switch modelService.state {
        case .ready, .generating:
            modelService.stopGeneration()
            modelService.unload()
        case .loading:
            break
        case .unloaded, .failed:
            modelService.startLoad()
        }
    }

    /// Welcoming conversation entry: seeds a friendly assistant greeting and focuses the composer.
    private func startWelcomingChat() {
        if case .unloaded = modelService.state {
            modelService.startLoad()
        }
        messages = [
            ChatMessage(
                role: .assistant,
                content: "هلا والله! أنا Sand2، جاهز أساعدك. وش تبي نسوي اليوم؟"
            )
        ]
        didGreet = true
        focused = true
    }

    private func send() {
        if isSending {
            modelService.stopGeneration()
            isSending = false
            return
        }

        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        if messages.isEmpty {
            startWelcomingChat()
        }

        if case .unloaded = modelService.state {
            modelService.startLoad()
        }

        let user = ChatMessage(role: .user, content: text)
        var assistant = ChatMessage(role: .assistant, content: "", isStreaming: true)
        messages.append(user)
        messages.append(assistant)
        draft = ""
        focused = false
        isSending = true

        let history = messages.filter { message in
            if message.id == assistant.id { return false }
            return true
        }
        let assistantID = assistant.id

        modelService.generate(
            messages: history,
            onToken: { token in
                Task { @MainActor in
                    guard let idx = messages.firstIndex(where: { $0.id == assistantID }) else { return }
                    messages[idx].content += token
                }
            },
            onComplete: { _ in
                Task { @MainActor in
                    if let idx = messages.firstIndex(where: { $0.id == assistantID }) {
                        messages[idx].isStreaming = false
                        if messages[idx].content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            messages[idx].content = "ما قدرت أرد الحين. تأكد إن الموديل محمّل من Hugging Face وحاول مرة ثانية."
                        }
                    }
                    isSending = false
                }
            }
        )
    }
}

import SwiftUI

/// First-run Arabic (Saudi) onboarding — two screens, one job each.
struct Sand2OnboardingView: View {
    @ObservedObject private var settings = AppSettings.shared
    @State private var page = 0

    var body: some View {
        ZStack {
            Sand2PageBackground()

            VStack(spacing: 0) {
                HStack {
                    Text("Sand2")
                        .font(.system(size: 34, weight: .bold, design: .rounded))
                        .foregroundStyle(Sand2Theme.ink)
                    Spacer()
                    if page == 0 {
                        Button("تخطّي") { finish() }
                            .font(.system(.subheadline, design: .rounded).weight(.semibold))
                            .foregroundStyle(Sand2Theme.inkMuted)
                            .frame(minHeight: 44)
                    }
                }
                .padding(.horizontal, Sand2Theme.pagePad)
                .padding(.top, 12)

                TabView(selection: $page) {
                    onboardingPage(
                        eyebrow: "على جهازك فقط",
                        title: "ذكاء اصطناعي\nبدون سحابة",
                        bodyText: "شغّل نماذج مفتوحة المصدر على الآيفون. محادثاتك وملفاتك تبقى عندك — ما في حساب ولا تتبّع.",
                        glyph: "lock.iphone"
                    )
                    .tag(0)

                    onboardingPage(
                        eyebrow: "من Hugging Face",
                        title: "حمّل الموديل\nوابدأ فوراً",
                        bodyText: "ابحث عن نموذج مناسب، نزّله مرّة واحدة، بعدين استخدمه أوفلاين. اخترنا لك نماذج خفيفة تناسب الجهاز.",
                        glyph: "arrow.down.circle.fill"
                    )
                    .tag(1)
                }
                .tabViewStyle(.page(indexDisplayMode: .never))

                VStack(spacing: 14) {
                    HStack(spacing: 6) {
                        ForEach(0..<2, id: \.self) { idx in
                            Capsule()
                                .fill(idx == page ? Sand2Theme.accent : Sand2Theme.ink.opacity(0.15))
                                .frame(width: idx == page ? 22 : 7, height: 7)
                        }
                    }

                    Sand2PrimaryButton(
                        title: page == 0 ? "التالي" : "ابدأ مع Sand2",
                        systemImage: page == 0 ? nil : "sparkles"
                    ) {
                        if page == 0 {
                            withAnimation(.easeInOut(duration: 0.22)) { page = 1 }
                        } else {
                            finish()
                        }
                    }
                }
                .padding(.horizontal, Sand2Theme.pagePad)
                .padding(.bottom, Sand2Theme.pagePad)
            }
        }
        .environment(\.layoutDirection, .rightToLeft)
        .environment(\.locale, Locale(identifier: "ar_SA"))
    }

    private func onboardingPage(
        eyebrow: String,
        title: String,
        bodyText: String,
        glyph: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            Spacer(minLength: 24)

            Image(systemName: glyph)
                .font(.system(size: 42, weight: .semibold))
                .foregroundStyle(Sand2Theme.accent)
                .frame(width: 72, height: 72)
                .background(Sand2Theme.accentSoft, in: RoundedRectangle(cornerRadius: 22, style: .continuous))

            Text(eyebrow)
                .font(.system(.caption, design: .rounded).weight(.bold))
                .foregroundStyle(Sand2Theme.accent)

            Text(title)
                .font(.system(size: 36, weight: .bold, design: .rounded))
                .foregroundStyle(Sand2Theme.ink)
                .lineSpacing(2)

            Text(bodyText)
                .font(.system(.body, design: .rounded))
                .foregroundStyle(Sand2Theme.inkMuted)
                .lineSpacing(4)

            Spacer()
        }
        .padding(.horizontal, Sand2Theme.pagePad)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func finish() {
        LocalizationService.shared.setLanguage(.arabic)
        withAnimation(.easeOut(duration: 0.2)) {
            settings.hasSeenOnboarding = true
        }
    }
}

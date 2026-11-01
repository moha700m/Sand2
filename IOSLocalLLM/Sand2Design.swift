import SwiftUI

/// Sand2 visual system — Arabic-first, UI/UX Pro style: one job per screen,
/// clear hierarchy, generous spacing, no decorative noise.
enum Sand2Theme {
    static let ink = Color(red: 0.07, green: 0.12, blue: 0.10)
    static let inkMuted = Color(red: 0.35, green: 0.42, blue: 0.38)
    static let accent = Color(red: 0.00, green: 0.42, blue: 0.21) // Saudi green
    static let accentSoft = Color(red: 0.00, green: 0.42, blue: 0.21).opacity(0.12)
    static let surface = Color.white.opacity(0.92)
    static let stroke = Color.black.opacity(0.06)
    static let danger = Color(red: 0.75, green: 0.18, blue: 0.18)
    static let good = Color(red: 0.12, green: 0.55, blue: 0.32)

    static let pagePad: CGFloat = 22
    static let gap: CGFloat = 14
    static let radius: CGFloat = 18
}

struct Sand2PageBackground: View {
    var body: some View {
        LinearGradient(
            colors: [
                Color(red: 0.95, green: 0.97, blue: 0.95),
                Color(red: 0.90, green: 0.94, blue: 0.91),
                Color(red: 0.86, green: 0.91, blue: 0.88)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
        .overlay {
            // Soft atmospheric wash — not flat, not noisy.
            RadialGradient(
                colors: [Sand2Theme.accent.opacity(0.10), .clear],
                center: .topTrailing,
                startRadius: 20,
                endRadius: 420
            )
            .ignoresSafeArea()
        }
    }
}

struct Sand2PrimaryButton: View {
    let title: String
    var systemImage: String? = nil
    var enabled: Bool = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if let systemImage {
                    Image(systemName: systemImage)
                }
                Text(title)
                    .font(.system(.headline, design: .rounded).weight(.semibold))
                Spacer(minLength: 0)
                Image(systemName: "arrow.left")
                    .font(.system(size: 14, weight: .semibold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .frame(maxWidth: .infinity, minHeight: 54)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(enabled ? Sand2Theme.accent : Sand2Theme.inkMuted.opacity(0.35))
            )
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.7)
    }
}

struct Sand2SecondaryButton: View {
    let title: String
    var systemImage: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if let systemImage {
                    Image(systemName: systemImage)
                }
                Text(title)
                    .font(.system(.subheadline, design: .rounded).weight(.semibold))
            }
            .foregroundStyle(Sand2Theme.ink)
            .padding(.horizontal, 16)
            .frame(minHeight: 46)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Sand2Theme.surface)
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .stroke(Sand2Theme.stroke, lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
    }
}

struct Sand2SectionLabel: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.system(.caption, design: .rounded).weight(.bold))
            .foregroundStyle(Sand2Theme.inkMuted)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

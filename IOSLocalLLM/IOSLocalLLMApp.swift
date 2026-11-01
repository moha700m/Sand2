import SwiftUI
import UIKit

@main
struct OnDeviceLASApp: App {
    @UIApplicationDelegateAdaptor(LASAppDelegate.self) private var appDelegate
    @ObservedObject private var settings = AppSettings.shared
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            Group {
                if settings.hasSeenOnboarding {
                    Sand2HomeView()
                        .transition(.opacity)
                } else {
                    Sand2OnboardingView()
                        .transition(.opacity)
                }
            }
            .environment(\.layoutDirection, .rightToLeft)
            .environment(\.locale, Locale(identifier: "ar_SA"))
        }
        .onChange(of: scenePhase) { _, newPhase in
            LASLifecycleController.shared.handle(newPhase)
        }
    }
}

final class LASAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        // Arabic (Saudi) first — before any UI mounts.
        LocalizationService.shared.setLanguage(.arabic)
        UserDefaults.standard.set("ar", forKey: "uiLanguage")
        CrashReporter.shared.install()
        Diagnostics.shared.breadcrumb("Sand2 launch", category: "lifecycle")
        return true
    }

    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        guard identifier == BackgroundDownloadCoordinator.sessionIdentifier else {
            completionHandler()
            return
        }
        RuntimeLogCenter.emit(
            "Reattaching background model-download session",
            subsystem: "download"
        )
        Task { @MainActor in
            let coordinator = BackgroundDownloadCoordinator.shared
            coordinator.systemCompletionHandler = completionHandler
            coordinator.reattach()
        }
    }
}

/// Lifecycle policy for the on-device product.
@MainActor
final class LASLifecycleController {
    static let shared = LASLifecycleController()

    private var epoch: UInt64 = 0

    func handle(_ phase: ScenePhase) {
        epoch &+= 1
        let currentEpoch = epoch
        Task { [weak self] in
            guard let self else { return }
            await self.apply(phase, epoch: currentEpoch)
        }
    }

    private func apply(_ phase: ScenePhase, epoch: UInt64) async {
        guard epoch == self.epoch else { return }

        switch phase {
        case .active:
            CrashReporter.shared.markRunning()
            guard AppSettings.shared.hasSeenOnboarding else { return }
            if AppSettings.shared.localAPIEnabled {
                await LocalAPIManager.shared.start()
            }
            guard epoch == self.epoch else { return }
            if AppSettings.shared.localAPIAutoLoadModel {
                CodingAssistantService.shared.startLoad()
            }
            guard epoch == self.epoch else { return }
        case .background:
            CrashReporter.shared.markCleanShutdown()
            await LocalAPIManager.shared.stop()
            guard epoch == self.epoch else { return }
            CodingAssistantService.shared.cancelLoad()
            CodingAssistantService.shared.stopGeneration()
            await CodingAssistantService.shared.unloadAndWaitForCleanup()
            guard epoch == self.epoch else { return }
            await MLXGenerationGate.shared.clearCacheWhenIdle()
        case .inactive:
            break
        @unknown default:
            break
        }
    }
}

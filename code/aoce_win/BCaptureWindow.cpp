#include "BCaptureWindow.hpp"

namespace aoce {
namespace win {

BCaptureWindow::BCaptureWindow(/* args */) {}

BCaptureWindow::~BCaptureWindow() {}

void BCaptureWindow::onObserverEvent(CaptureEventId eventId, LogLevel level,
                                     const char* msg) {
    if (observer) {
        observer->onEvent(eventId, level, msg);
    }
}

void BCaptureWindow::setObserver(ICaptureObserver* observer) {
    this->observer = observer;
}

bool BCaptureWindow::bCapturing() { return bCapture; }

}  // namespace win
}  // namespace aoce
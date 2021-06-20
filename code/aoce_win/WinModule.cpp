#include "WinModule.hpp"

#include <AoceManager.hpp>

#include "window/MCaptureWindow.hpp"
#include "window/MWindowManager.hpp"

namespace aoce {
namespace win {

WinModule::WinModule(/* args */) {}

WinModule::~WinModule() {}

bool WinModule::loadModule() {
    AoceManager::Get().addIWindowManager(WindowType::win, new MWindowManager());
    AoceManager::Get().addICaptureWindow(CaptureType::win_bitblt,
                                         new MCaptureWindow());
    return true;
}

void WinModule::unloadModule() {
    AoceManager::Get().removeIWindowManager(WindowType::win);
    AoceManager::Get().removeICaptureWindow(CaptureType::win_bitblt);
}

ADD_MODULE(WinModule, aoce_win)

}  // namespace win
}  // namespace aoce
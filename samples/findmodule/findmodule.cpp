#include <AoceManager.hpp>
#include <Module/ModuleManager.hpp>
#include <iostream>
#include <string>
using namespace aoce;

int main() {
    std::cout << "start" << std::endl;

    int a;
    ModuleManager::Get().regAndLoad("aoce_win_mf");
    auto& deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << deviceList.size() << std::endl;
    for (const auto& video : deviceList) {
        wchar_t* pname = (wchar_t*)video->getName().data();
        std::wstring name(pname);
        std::wstring id((wchar_t*)video->getId().data());
        std::wcout << "name: " << name << std::endl;
        std::wcout << "id: " << id << std::endl;
    }
    ModuleManager::Get().unloadModule("aoce_win_mf");

    // ModuleManager::Get().regAndLoad("aoce_win_mf");
    // deviceList =
    //     AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << deviceList.size() << std::endl;
    for (const auto& video : deviceList) {
        wchar_t* pname = (wchar_t*)video->getName().data();
        std::wstring name(pname);
        std::wstring id((wchar_t*)video->getId().data());
        std::wcout << "name: " << name << std::endl;
        std::wcout << "id: " << id << std::endl;
    }
    // ModuleManager::Get().unloadModule("aoce_win_mf");

    std::cin >> a;
}
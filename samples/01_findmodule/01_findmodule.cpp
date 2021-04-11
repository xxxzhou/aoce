#include <AoceManager.hpp>
#include <Module/ModuleManager.hpp>
#include <iostream>
#include <string>
using namespace aoce;

int main() {
    std::cout << "start" << std::endl;

    int a;
    ModuleManager::Get().regAndLoad("aoce_win_mf");
    const auto& deviceList =
        AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << deviceList.size() << std::endl;
    for (const auto& video : deviceList) {
        std::string name = video->getName();
        std::string id = video->getId();
        std::cout << "name: " << name << std::endl;
        std::cout << "id: " << id << std::endl;
    }
    ModuleManager::Get().unloadModule("aoce_win_mf");

    // 验证再次加载
    // ModuleManager::Get().regAndLoad("aoce_win_mf");
    // deviceList =
    //     AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    std::cout << deviceList.size() << std::endl;
    for (const auto& video : deviceList) {
        std::string name = video->getName();
        std::string id = video->getId();
        std::cout << "name: " << name << std::endl;
        std::cout << "id: " << id << std::endl;
    }
    // ModuleManager::Get().unloadModule("aoce_win_mf");

    std::cin >> a;
}
#include <AoceManager.hpp>
#include <iostream>
#include <module/ModuleManager.hpp>
#include <string>
#include <vector>
using namespace aoce;

int main() {
    int a;
    std::cout << "start load module test." << std::endl;

    loadAoce();
    std::vector<std::string> models = {
        "aoce_vulkan", "aoce_vulkan_extra", "aoce_win",
        "aoce_winrt",  "aoce_win_mf",       "aoce_cuda",
        "aoce_agora",  "aoce_talkto",       "aoce_talkto_cuda"};
    std::cout << "---check failed module resutl" << std::endl;
    for (const auto& name : models) {
        if (!checkLoadModel(name.c_str())) {
            std::string msg = name + " not load.";
            logMessage(LogLevel::warn, msg);
        }
    }
    std::cout << "---end resutl" << std::endl;
    unloadAoce();
    std::cout << "end load module test." << std::endl;
    std::cin >> a;
    // int a;
    // ModuleManager::Get().regAndLoad("aoce_win_mf");
    // const auto& deviceList =
    //     AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    // std::cout << deviceList.size() << std::endl;
    // for (const auto& video : deviceList) {
    //     std::string name = video->getName();
    //     std::string id = video->getId();
    //     std::cout << "name: " << name << std::endl;
    //     std::cout << "id: " << id << std::endl;
    // }
    // ModuleManager::Get().unloadModule("aoce_win_mf");

    // // 验证再次加载
    // // ModuleManager::Get().regAndLoad("aoce_win_mf");
    // // deviceList =
    // //
    // AoceManager::Get().getVideoManager(CameraType::win_mf)->getDeviceList();
    // std::cout << deviceList.size() << std::endl;
    // for (const auto& video : deviceList) {
    //     std::string name = video->getName();
    //     std::string id = video->getId();
    //     std::cout << "name: " << name << std::endl;
    //     std::cout << "id: " << id << std::endl;
    // }
    // // ModuleManager::Get().unloadModule("aoce_win_mf");

    // std::cin >> a;
}
#include "AoceManager.hpp"

namespace aoce {

AoceManager* AoceManager::instance = nullptr;
AoceManager& AoceManager::Get() {
    if (instance == nullptr) {
        instance = new AoceManager();
    }
    return *instance;
}

AoceManager::AoceManager(/* args */) {}

AoceManager::~AoceManager() {}

}  // namespace aoce
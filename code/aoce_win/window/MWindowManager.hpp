#pragma once
#include "../WinHelp.hpp"
#include "MWindow.hpp"

namespace aoce {
namespace win {

class MWindowManager : public IWindowManager {
   private:
    /* data */
    std::vector<std::unique_ptr<MWindow>> windows;

   public:
    MWindowManager(/* args */);
    virtual ~MWindowManager();

   public:
    void getWindowList();
    void addWindow(HWND hwnd);

   public:
    virtual int32_t getWindowCount(bool bUpdate = false) override;
    virtual IWindow* getWindow(int32_t index) override;
    virtual IWindow* getDesktop() override;
};

}  // namespace win
}  // namespace aoce
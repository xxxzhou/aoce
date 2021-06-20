#pragma once
#include "../WinHelp.hpp"

namespace aoce {
namespace win {

class MWindow : public IWindow {
   private:
    /* data */
    HWND hwnd = nullptr;
    std::string title = "";
    uint64_t pId = 0;
    uint64_t tId = 0;

   public:
    MWindow(/* args */);
    ~MWindow();

   public:
    void initWindow(HWND hd);

   public:
    virtual const char* getTitle() override;
    virtual void* getHwnd() override;
    virtual uint64_t getProcessId() override;
    virtual uint64_t getMainThreadId() override;

    virtual bool bValid() override;
};

}  // namespace win
}  // namespace aoce
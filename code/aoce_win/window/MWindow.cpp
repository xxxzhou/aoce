#include "MWindow.hpp"

namespace aoce {
namespace win {

MWindow::MWindow(/* args */) {}

MWindow::~MWindow() {}

void MWindow::initWindow(HWND hd) {
    hwnd = hd;
    int32_t len = ::GetWindowTextLengthA(hwnd);
    std::vector<char> temp(len + 1);
    if (len > 0) {
        ::GetWindowTextA(hwnd, temp.data(), len + 1);
        title = temp.data();
    }
    unsigned long pId = 0;
    //线程id
    unsigned long tId = 0;
    tId = ::GetWindowThreadProcessId(hwnd, &pId);
    this->pId = pId;
    this->tId = tId;
}

const char* MWindow::getTitle() { return title.c_str(); }

void* MWindow::getHwnd() { return hwnd; }

uint64_t MWindow::getProcessId() { return pId; }

uint64_t MWindow::getMainThreadId() { return tId; }

bool MWindow::bValid() { return validWindow(hwnd); }

}  // namespace win
}  // namespace aoce
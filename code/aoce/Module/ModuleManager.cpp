#include "ModuleManager.hpp"

#include "../AoceManager.hpp"

#if WIN32
#include <Shlwapi.h>
#include <Windows.h>
#include <dbghelp.h>
#include <sysinfoapi.h>
#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "shlwapi.lib")
#elif __ANDROID__
#include <dlfcn.h>
#endif

namespace aoce {

ModuleManager* ModuleManager::instance = nullptr;
ModuleManager& ModuleManager::Get() {
    if (instance == nullptr) {
        instance = new ModuleManager();
    }
    return *instance;
}

ModuleManager::ModuleManager(/* args */) {}

ModuleManager::~ModuleManager() {}

void ModuleManager::registerModule(const char* name,
                                   loadModuleHandle handle) {
    if (modules.find(name) != modules.end()) {
        return;
    }
    ModuleInfo* moduleInfo = new ModuleInfo();
    modules[name] = moduleInfo;
    moduleInfo->name = name;
#if __ANDROID__
    moduleInfo->name = "lib" + moduleInfo->name + ".so";
#endif
    moduleInfo->onLoadEvent = handle;
}

void ModuleManager::loadModule(const char* name) {
    if (modules.find(name) == modules.end()) {
        return;
    }
    ModuleInfo* moduleInfo = modules[name];
    if (moduleInfo->load) {
        return;
    }
    if (moduleInfo->onLoadEvent) {
        moduleInfo->module = moduleInfo->onLoadEvent();
    } else {
        loadModuleAction loadAction = nullptr;
#if WIN32
        char temp[512] = {0};
        GetDllDirectoryA(512, temp);
        char sz[512] = {0};
        HMODULE ihdll = GetModuleHandleA("aoce.dll");
        ::GetModuleFileNameA(ihdll, sz, 512);
        ::PathRemoveFileSpecA(sz);
        SetDllDirectoryA(sz);
        moduleInfo->handle = LoadLibraryExA(moduleInfo->name.c_str(), nullptr,
                                            LOAD_WITH_ALTERED_SEARCH_PATH);
        if (moduleInfo->handle) {
            loadAction = (loadModuleAction)GetProcAddress(
                (HMODULE)moduleInfo->handle, "NewModule");
        }
        SetDllDirectoryA(temp);
#elif __ANDROID__
        moduleInfo->handle =
            dlopen(moduleInfo->name.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (moduleInfo->handle) {
            loadAction =
                (loadModuleAction)dlsym(moduleInfo->handle, "NewModule");
        }
#endif
        // 检查是否找到dll
        if (!moduleInfo->handle) {
            logMessage(LogLevel::warn, moduleInfo->name + ": dll load failed.");
            return;
        }
        // 检查是否加载module
        if (loadAction) {
            moduleInfo->module = loadAction();
        } else {
            logMessage(LogLevel::warn,
                       moduleInfo->name + " is load,but no find method NewModule.");
            return;
        }
    }
    if (!moduleInfo->module) {
        logMessage(LogLevel::warn, moduleInfo->name + ": init module failed.");
    }
    // 调用注册
    if (moduleInfo->module) {
        moduleInfo->load = moduleInfo->module->loadModule();
        if (moduleInfo->load) {
            logMessage(LogLevel::info, moduleInfo->name + ": regedit module success.");
        } else {
            logMessage(LogLevel::warn, moduleInfo->name + ": regedit module failed.");
        }
    }
}

void ModuleManager::regAndLoad(const char* name) {
    registerModule(name);
    loadModule(name);
}

void ModuleManager::unloadModule(const char* name) {
    if (modules.find(name) == modules.end()) {
        return;
    }
    ModuleInfo* moduleInfo = modules[name];
    if (!moduleInfo->load) {
        return;
    }
    if (moduleInfo->module) {
        moduleInfo->module->unloadModule();
        delete moduleInfo->module;
        moduleInfo->module = nullptr;
    }
    if (moduleInfo->handle) {
#if WIN32
        FreeLibrary((HMODULE)moduleInfo->handle);
#elif __ANDROID__
        dlclose(moduleInfo->handle);
#endif
        moduleInfo->handle = nullptr;
    }
    moduleInfo->load = false;
}

}  // namespace aoce

#if WIN32

LONG WINAPI unhandledFilter(struct _EXCEPTION_POINTERS* lpExceptionInfo) {
    LONG ret = EXCEPTION_EXECUTE_HANDLER;
    TCHAR szFileName[64];
    SYSTEMTIME st;
    ::GetLocalTime(&st);
    wsprintf(szFileName, TEXT("OEIP_%04d%02d%02d-%02d%02d%02d-%ld-%ld.dmp"),
             st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond,
             GetCurrentProcessId(), GetCurrentThreadId());

    HANDLE hFile = ::CreateFile(szFileName, GENERIC_WRITE, 0, NULL,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION ExInfo;
        ExInfo.ThreadId = ::GetCurrentThreadId();
        ExInfo.ExceptionPointers = lpExceptionInfo;
        ExInfo.ClientPointers = false;

        // write the dump
        BOOL bOK =
            MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile,
                              MiniDumpNormal, &ExInfo, NULL, NULL);
        ::CloseHandle(hFile);
    }
    return ret;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD dwReason, LPVOID lpReserved) {
    if (dwReason == DLL_PROCESS_ATTACH) {
        SetUnhandledExceptionFilter(
            (LPTOP_LEVEL_EXCEPTION_FILTER)unhandledFilter);
        // aoce::ModuleManager::Get().registerModule("aoce-mf");
    } else if (dwReason == DLL_PROCESS_DETACH) {
    }
    return TRUE;
}

#elif __ANDROID__

// static bool bAttach = false;
// jint JNI_OnLoad(JavaVM* jvm, void*) {
//    aoce::AndroidEnv androidEnv = {};
//    androidEnv.vm = jvm;
//    // aoce::AoceManager::Get().initAndroid(androidEnv);
//}
//
// void JNI_OnUnload(JavaVM* jvm, void*) {
//    aoce::AoceManager::Get().detachThread();
//}
#endif
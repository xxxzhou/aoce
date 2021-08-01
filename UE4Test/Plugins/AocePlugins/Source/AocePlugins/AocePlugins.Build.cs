// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class AocePlugins : ModuleRules
{
    private string suffix = ".lib";
    public AocePlugins(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicIncludePaths.AddRange(
            new string[] {
				// ... add public include paths required here ...
            }
            );


        PrivateIncludePaths.AddRange(
            new string[] {
				// ... add other private include paths required here ...
			}
            );


        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core","CoreUObject","Projects","Engine","RHI", "RenderCore", "UMG", "SlateCore", "OpenGLDrv","Json","JsonUtilities",
				// ... add other public dependencies that you statically link with here ...
			}
            );


        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "CoreUObject",
                "Engine",
                "Slate",
                "SlateCore",
				// ... add private dependencies that you statically link with here ...	
			}
            );


        DynamicallyLoadedModuleNames.AddRange(
            new string[]
            {
				// ... add any modules that your module loads dynamically here ...
			}
            );
        CppStandard = CppStandardVersion.Cpp14;
        // 头文件
        PublicSystemIncludePaths.AddRange(new string[] { Path.Combine(ThirdPartyPath, "Aoce/win/include") });
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            PublicDefinitions.Add("__ANDROID__=0");
            PublicDefinitions.Add("WIN32=1");
            string aocePath = Path.Combine(ThirdPartyPath, "Aoce/win");
            string libPath = Path.Combine(aocePath, "lib");
            string binPath = Path.Combine(aocePath, "bin");
            // 链接库
            PublicSystemLibraryPaths.Add(libPath);
            string[] runLibs = { "aoce", "aoce_talkto_cuda", "aoce_talkto", "aoce_vulkan_extra" };
            foreach (var runLib in runLibs)
            {
                PublicSystemLibraries.Add(runLib + ".lib");
                // 需要延迟加载的dll,不加的话,需把相应dll拷贝到工程的Binaries,否则编辑器到75%就因加载不了dll crash.     
                PublicDelayLoadDLLs.Add(runLib + ".dll");
            }
            foreach (string path in Directory.GetFiles(binPath, "*.*", SearchOption.AllDirectories))
            {
                RuntimeDependencies.Add(path);
            }
        }
        else if (Target.Platform == UnrealTargetPlatform.Android)
        {
            // PublicDependencyModuleNames.Add("OpenGLDrv");
            suffix = ".so";
            Definitions.Add("__ANDROID__=1");
            Definitions.Add("WIN32=0");
            AdditionalPropertiesForReceipt.Add("AndroidPlugin", Path.Combine(ModuleDirectory, "AocePlugins_APL.xml"));
            string aocePath = Path.Combine(ThirdPartyPath, "Aoce/android");
            // armeabi-v7a / arm64-v8a  Architecture
            string[] runAbis = { "arm64-v8a" };
            foreach (var runAbi in runAbis)
            {
                string libPath = Path.Combine(aocePath, runAbi);
                // 链接库
                PublicAdditionalLibraries.Add(Path.Combine(libPath, "libaoce.so"));
                PublicAdditionalLibraries.Add(Path.Combine(libPath, "libaoce_talkto.so"));
            }
        }
    }

    private string ThirdPartyPath
    {
        get { return Path.GetFullPath(Path.Combine(ModuleDirectory, "../../ThirdParty/")); }
    }
}

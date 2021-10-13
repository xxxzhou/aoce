# UE4整理

## 工具

用UE4 Rider替换VS,C++相关开发提升一个量级.

## LatentActionManager

[UE4关卡流式加载与LatentAction](https://zhuanlan.zhihu.com/p/368055153)

用来管理UObject上的各个Action,主要结构map,key对应UObject,value对应FPendingLatentAction列表(FActionList),FPendingLatentAction方便外部用一个UUID(int32)来表示,所以需要自己维护一个FPendingLatentAction对应UUID的关系,相应UObject/UUID封装结果使用FLatentActionInfo表示.

## 宏

Blueprintable/NotBlueprintable 此类公开/不公开为创建蓝图的可接受基类.

BlueprintType 将此类公开为可用于蓝图中的变量的类型.

## 编辑器扩展

[UE4编辑器开发（三）资源类型拓展](https://zhuanlan.zhihu.com/p/135315547)

[UE4项目记录（十一）自定义资源及](https://zhuanlan.zhihu.com/p/41332747)

## 蓝图泛形

[第3期 ue4 泛型蓝图节点的实现及应用实例](https://zhuanlan.zhihu.com/p/148209184)

[UE4中蓝图函数的泛型](https://zhuanlan.zhihu.com/p/144301168)

## 行为树

[UE4行为树详解-持续更新](https://zhuanlan.zhihu.com/p/143298443)

[浅析UE4-BehaviorTree的特性](https://zhuanlan.zhihu.com/p/139514376)

## 异步操作

[将异步操作封装为蓝图节点](https://zhuanlan.zhihu.com/p/107021667)

## 智能指针

[虚幻4：智能指针基础](https://zhuanlan.zhihu.com/p/94198883)

## 代理

[UE4中的代理(Delegate)使用总结](https://zhuanlan.zhihu.com/p/126630820)

C++之间传递最好使用基本代理,蓝图使用动态代理,如果要把C++给蓝图使用,加个转接函数.

``` c++

DECLARE_DELEGATE_OneParam(FDownResultDelegate, bool /*bSuccess*/);
DECLARE_DYNAMIC_DELEGATE_OneParam(FDownResultBPDelegate, bool, bSuccess);
void UAudioBPLibrary::RequestAudioFile(const FTextAudio& AudioText, int32 Speed, FDownResultBPDelegate onBPResult)
{
	FDownResultDelegate onResult = FDownResultDelegate::CreateLambda([onBPResult](bool bSucess)
	{
		onBPResult.ExecuteIfBound(bSucess);
	});
	RequestAudioFile(AudioText, Speed, onResult);
}

void UAudioBPLibrary::RequestAudioFile(const FTextAudio& AudioText, int32 Speed, FDownResultDelegate onResult)
{
    ...
}
```

# Oculus quest开发注意事项

我接手的Oculus quest，里面使用的Facebook账号已经被封停，导致里面所有功能全部不能正常使用，并且也没有提供切换账号的功能。不得已下，重置成出产设置。所以我算是拿一个新的从头开始配置。

## 配置

1 Oculus quest需要在手机下Oculus app,需要翻墙.

2 Oculus quest在2023年之前可以使用Oculus账号(现好像已经关闭注册),Oculus quest推荐你使用facebook账号,但是国内的facebook容易被封停，如何封停，容易导致Oculus quest设备时所有选项都无法正常使用，需要重置成出产设置，非常麻烦，个人建议使用Oculus账号，如果你有的话。

3 Oculus quest需要进行配对这些设置，其中WIFI比较麻烦，需要可以翻墙，如果你有可以翻墙的路由器，那就没什么说的了，如果没有，在没进Oculus quest系统界面前的配置界面，是不能设置代理的，所以就算你电脑/手机可以翻墙，分享的热点是翻不了的，需要SSTAP将SSR翻墙节点转换成类VPN的全局效果，然后用Connectify Hotspot把此链接变成WIFI热点，这样Oculus quest能直连此热点而不设置代理就可以翻墙，大约需要更新1G左右的内容，要等一久。

4 更新完成后，可以改成前面分享的热点WIFI,设置代理一样可以翻墙了。

5 开发者选项需要在手机Oculus app启用，然后按照提示去相应网站执行，然后连接Oculus quest与电脑应该就会提示是否Debug,如果没有提示，一是看是否完成设置，二是重启Oculus quest.

## 事项

1 Oculus quest 的android api = 25.

2 [UE4开发配置](https://developer.oculus.com/documentation/unreal/unreal-quick-start-guide-quest/)

3 注意相应的AndroidManifest.xml添加如下二行

```xml
<uses-feature android:name="android.hardware.usb.host" />
<uses-feature android:name="android.hardware.vr.headtracking" android:version="1" android:required="true" />
```

[Android Manifest Settings](https://developer.oculus.com/documentation/native/android/mobile-native-manifest)

在UE4里就是项目设置/平台/Android/高级APK打包/Oculus移动设备的包勾选上

## 测试

1080P下,显示出来直接卡死,CPU占满,此时GPU占用相反很低,分析应该是1080P把IO带宽全部占满.
2K 推流后,手机端拉流显示只有1080P,后续与agora确认。

性能部分，有三个点我认为可以优化的部分，分别是解码，GPU计算，给UE4显示。

从下表可以看到，解码主要使用CPU，GPU计算部分影响UE4呈现桢率，计算交换UE4显示主要影响CPU，当分辨率为1080P，游戏与声网很大可能因其带宽(CPU-内存)占满，导致性能直接崩溃成1FPS。

1. 解码末计算: 只包含解码占用。
2. 计算末显示: 包含解码,vulkan gpu计算。
3. 计算且显示: 包含解码,vulkan gpu计算,计算结果与UE4交互呈现。

简单场景(FPS/CPU占用/GPU占用)

|分辨率|末解码末计算|解码末计算|计算末显示|计算且显示|
|---|---|---|---|---|
|720P|(50-61)(28-40)(98)|(50-61)(55-75)(97)|(41)(50-70)(90-99)|(33)(80-100)(90-98)|
|1080P|(50-61)(28-40)(98)|(50-61)(65-100)(97)|(30)(80-100)(98)|(1-3)(100)(52)|

复杂场景(UE4太阳神庙)

|分辨率|末解码末计算|解码末计算|计算末显示|计算且显示|
|---|---|---|---|---|
|720P|(20-30)(20-40)(97)|(20-30)(40-60)(97)|(23)(44-65)(90-99)|(22-26)(60-80)(96)|
|1080P|(20-30)(20-40)(97)|(20-30)(70-100)(97)|(20)(90-100)(98)|(1)(100)(60)|

## 需要解决的问题

测试720,现测试场景很简单的情况,末集成框架功能平均55fps左右,而集成框架显示拉流33fps.复杂场景的情况下，末集成框架平均24fps左右,集成框架20fps.

1 延迟过高,1-2s.

2 需要测试1080P/2K相应的效果.

3 桢率不高的问题.

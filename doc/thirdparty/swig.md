# swig

[testswig](https://github.com/OlegJakushkin/TestSWIG)

[dotnet-native](https://github.com/Mizux/dotnet-native)

[swig中文手册](https://www.cnblogs.com/xuruilong100/tag/SWIG%203%20%E4%B8%AD%E6%96%87%E6%89%8B%E5%86%8C/)

[swig pdf](http://www.swig.org/Doc4.0/SWIGDocumentation.pdf)

[swig html](http://www.swig.org/Doc4.0/Sections.html#Sections)

[swig github](https://github.com/swig/swig/blob/master/Lib/csharp/typemaps.i)

[swig typemap](http://www.swig.org/Doc4.0/Typemaps.html#Typemaps)

[swig 高级用法代码](https://forge.naos-cluster.tech/aquinetic/f2i-consulting/fesapi/-/blob/c0a52292680e4ec316d2e3447b52f365a54cc400/cmake/swigModule.i)

## 问题

1. 回调类如何用C#的类继承并可以处理逻辑.
2. 继承的模板具体类后,通过swig转换丢失继承信息.
3. 一些参数如int [],uint8*,void*并没有转换成我想要的C#类型.

## 注意

带前缀SWIGTYPE_p的表明对应只是生成指针,没有对应属性生成.

如函数指针logEventAction就会生成SWIGTYPE_p_logEventAction类型,对我们来说,意思就不大.

1 IVideoManager::getDevices 返回IVideoDevice**,也会生成SWIGTYPE_p_p无意义,ignore忽略,并添加方法getDevice直接返回一个IVideoDevice指针给外部语言使用.

2 如上函数指针logEventAction,封装成一个回调Abserver类.

3 int32_t,uint32_t对应正确的C#类型,添加 %include "stdint.i".

```c++
%typemap(ctype)  void * "void *"
%typemap(imtype) void * "IntPtr"
// 针对CShpar,void* 转 intptr
%typemap(cstype) void * "IntPtr"
%typemap(csin)   void * "$csinput"
// 输入,C#类型转C++
%typemap(in)     void * %{ $1 = $input; %}
// 输出,C++转C#类型
%typemap(out)    void * %{ $result = $1; %}
%typemap(csout)  void * { return $imcall; }
```

回调abserver放最前面.

如果类有中继承具体化模板的类,需要在%include之前%template 模板,否则不能生成正确的继承关系.

实现方法类似为回调abserver类的每个方法对应声明对应函数指针对象,然后把C#对应override方法传入在对应每个函数指针.在上个项目Oeip里的Oeip_live里OeipLiveBackWrapper类用的就是这种方式.在win端也可通过COM技术实现C#与C++类之间互相继承交流,类似Oeip里的OeipLiveCom.

4 C++中的向下转换到别的语言向下转换,不得行.

[Java_adding_downcasts](http://www.swig.org/Doc3.0/Java.html#Java_adding_downcasts)

## SWIG

swig在处理里添加特有的宏(#ifdef SWIG).

[jni-faiss](https://github.com/gameofdimension/jni-faiss/blob/master/jni/swigfaiss.swig)不同语言处理.#ifdef SWIGJAVA/#ifdef SWIGPYTHON/#ifdef SWIGLUA

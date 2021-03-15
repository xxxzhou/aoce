# GLSL单独分离出来

[glslangValidator](http://manpages.ubuntu.com/manpages/focal/en/man1/glslangValidator.1.html)

考虑可能有100多种效果,glsl对应不同条件生成可能超过100多个文件,所以建立一个单独文件夹管理所有glsl文件.

glslindex.txt 文件用来指示如何需要编译的glsl文件目录.

compileglsl.py 用来针对glslindex.txt编译glsl文件.

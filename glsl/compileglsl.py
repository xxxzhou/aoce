import sys
import os
import glob
import subprocess
import shutil

path = os.path.dirname(os.path.realpath(__file__))
path = path.replace('\\', '/')

class fileNode():
    def __init__(self,srcfile,desfile,macro):
        self.srcfile = srcfile
        self.desfile = desfile
        self.macro  = macro

files = []
# glslindexcurrent glslindex glsl_tt
glslIndex = 'glslindexcurrent.txt'
with open(os.path.join(path, glslIndex), 'r') as lines:
    for line in lines:
        nodes = line.strip().split(" ")
        desNode = nodes[0]
        macro = ""
        lenNode = len(nodes)
        if lenNode < 1:
            continue
        if lenNode > 1:
            desNode = nodes[1]
        if lenNode > 2:
            for i in range(2,lenNode):
                macro += "-D" + nodes[i].strip() + " " 
        srcfile = os.path.join(path, 'source',nodes[0])   
        desfile = os.path.join(path, 'target',desNode+".spv")  
        fn = fileNode(srcfile,desfile,macro)    
        files.append(fn)

bSucess = True
for gfile in files:
    print("\n-------- %s --------\n" % gfile.desfile)    
    if len(gfile.macro) == 0:
        cmdStr = "glslangValidator -V %s -o %s" % (gfile.srcfile, gfile.desfile)
    else:
        cmdStr = "glslangValidator -V %s -o %s %s" % (gfile.srcfile, gfile.desfile, gfile.macro)
    if subprocess.call(cmdStr, shell=True) != 0 :
        bSucess = False
        break

print("\n-------- Compilation result --------\n")

def copyDir(dict):
    if not os.path.exists(dict):
        os.makedirs(dict)
    for gfile in files:
        shutil.copy(gfile.desfile,dict+os.path.basename(gfile.desfile))

if bSucess:
    print("SUCCESS: All %i shaders compiled to SPIR-V" % len(files))
    debugDict = os.path.join(path, "../build/bin/Debug/glsl/")
    releaseDict = os.path.join(path, "../build/bin/Debug/glsl/")
    installDict = os.path.join(path, "../build/install/win/bin/glsl/")
    # 复制到DEBUG目录
    copyDir(debugDict)
    # 复制到RELEASE目录 
    copyDir(releaseDict)
    # 复制到安装目录下
    copyDir(installDict)
    print("SUCCESS: Copy All shaders: " + glslIndex)
else:
	print("ERROR: %s not be compiled:\n" % glslIndex)    
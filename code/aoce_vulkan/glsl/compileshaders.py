import sys
import os
import glob
import subprocess
import shutil

path = os.path.dirname(os.path.realpath(__file__))
path = path.replace('\\', '/')

if len(sys.argv) > 2:
	if os.path.exists(sys.argv[1]):
		# sys.exit("%s is not a valid directory" % sys.argv[1])
		path = sys.argv[1]

shaderfiles = []
for exts in ('*.vert', '*.frag', '*.comp', '*.geom', '*.tesc', '*.tese'):
	shaderfiles.extend(glob.glob(os.path.join(path, exts))) 

failedshaders = []
for shaderfile in shaderfiles:
		print("\n-------- %s --------\n" % shaderfile)
		if subprocess.call("glslangValidator -V %s -o %s.spv" % (shaderfile, shaderfile), shell=True) != 0:
			failedshaders.append(shaderfile)

print("\n-------- Compilation result --------\n")

if len(failedshaders) == 0:
	print("SUCCESS: All shaders compiled to SPIR-V")
	dest = path + "/../../../build/bin/Debug/glsl/"
	risvfiles = []
	risvfiles.extend(glob.glob(os.path.join(path, '*.spv'))) 
	for risvfile in risvfiles:
		shutil.copy(risvfile,dest+os.path.basename(risvfile))
	dest = path + "/../../../build/bin/Release/glsl/"
	for risvfile in risvfiles:
		shutil.copy(risvfile,dest+os.path.basename(risvfile))
	print("SUCCESS: Copy All shaders: "+dest)
else:
	print("ERROR: %d shader(s) could not be compiled:\n" % len(failedshaders))
	for failedshader in failedshaders:
		print("\t" + failedshader)

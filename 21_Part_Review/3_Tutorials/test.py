import os

output = os.popen('nvidia-smi')

print(output.read())
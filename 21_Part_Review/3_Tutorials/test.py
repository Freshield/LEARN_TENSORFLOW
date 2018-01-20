import os

output = os.popen('nvidia-smi')

print(output.read())

a = []
a.append(1)
a.append(2)

print(a)

b = []
b.append(3)
print(b)
b = b + a
print(b)
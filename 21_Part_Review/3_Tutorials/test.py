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


#创建一个空类，底下把各种资料都存到类中
class PARAMETERS(object):
    pass
PARA = PARAMETERS()

BATCH_SIZE = 128
PARA.BATCH_SIZE = BATCH_SIZE

REG = 0.0001
PARA.REG = REG

EPOCH_NUM = 5
PARA.EPOCH_NUM = EPOCH_NUM

print(PARA.BATCH_SIZE)


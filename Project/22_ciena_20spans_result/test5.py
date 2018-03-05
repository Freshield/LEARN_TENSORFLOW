import math

def func(x):
    res = x ** 3 - 7 * (x ** 2) - 3 * x + 3
    return res

print func((14 - math.sqrt(232)) / 6)
print func(5)
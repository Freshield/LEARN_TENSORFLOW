a = [1,2]

print a

def test(para):
    temp = [2,3]
    for i in range(len(temp)):
        para[i] = temp[i]

test(a)

print a

[c,d] = a

print c
print d

e = 1
f = 2
g = [e,f]
test(g)
[e,f] = g
print g
print e
print f

print ''

i = 0
while i < 5:
    print i
    i = 0
    i += 1


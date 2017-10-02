import threading
from time import ctime,sleep

def music(func):
    for i in range(2):
        print "I was listening to %s. %s" % (func, ctime())
        sleep(1)

def movie(func):
    for i in range(2):
        print "I was at the %s! %s" % (func, ctime())
        sleep(5)

music('love sale')
movie('avator')
print 'all over %s' % ctime()
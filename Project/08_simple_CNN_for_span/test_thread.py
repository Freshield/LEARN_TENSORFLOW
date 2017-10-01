import threading
from time import ctime,sleep

def music(func):
    for i in range(2):
        print "I was listening to %s. %s" % (func,ctime())
        sleep(4)

def movie(func):
    for i in range(2):
        print "I was at the %s! %s" % (func, ctime())
        sleep(5)

threads = []
t1 = threading.Thread(target=music, args=('love',))
threads.append(t1)
t2 = threading.Thread(target=movie, args=('avater',))
threads.append(t2)
for t in threads:
    t.setDaemon(True)
    t.start()
t.join()
print "all over %s" % ctime()
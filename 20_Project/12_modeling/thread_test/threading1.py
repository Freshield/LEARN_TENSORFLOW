import threading
from time import ctime, sleep

def music(func):
    for i in range(2):
        print 'I was listening to %s. %s' % (func, ctime())
        sleep(1)

def movie(func):
    for i in range(2):
        print 'I was at the %s! %s' % (func, ctime())
        sleep(5)

threads = []
t1 = threading.Thread(target=music, args=('love sales',))
threads.append(t1)
t2 = threading.Thread(target=movie, args=('avator',))
threads.append(t2)

for t in threads:
    t.setDaemon(True)
    t.start()
    t.join()

print 'all over %s' % ctime()
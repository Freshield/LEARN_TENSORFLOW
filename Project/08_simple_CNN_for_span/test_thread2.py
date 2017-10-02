from time import sleep, ctime

import threading

def music(func):
    for i in range(2):
        print 'start playing: %s! %s' % (func, ctime())
        sleep(2)

def movie(func):
    for i in range(2):
        print 'start playing: %s! %s' % (func, ctime())
        sleep(5)

def player(name):
    r = name.split('.')[1]
    if r == 'mp3':
        music(name)
    else:
        if r == 'mp4':
            movie(name)
        else:
            print 'error'

list = ['love.mp3', 'avater.mp4']

threads = []
files = range(len(list))

for i in files:
    t = threading.Thread(target=player, args=(list[i],))
    threads.append(t)


for i in files:
    threads[i].start()

for i in files:
    threads[i].join()

print 'end:%s' % ctime()
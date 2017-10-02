import Queue
import threading
import time

exitFlag = 0

class myThread(threading.Thread):
    def __init__(self,threadID,name,q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print 'starting ' + self.name
        process_data(self.name,self.q)
        print 'exiting ' + self.name

#how to process the data
def process_data(threadName, q):
    while not exitFlag:
        #get the lock
        queueLock.acquire()
        #check the queue status
        if not workQueue.empty():
            data = q.get()
            queueLock.release()
            print '%s processing %s, %s' % (threadName,data,time.ctime())
            time.sleep(3)
        else:
            queueLock.release()
            time.sleep(3)


#init
threadList = ['Thread-1','Thread-2','Thread-3']
nameList = ['One','Two','Three','Four',"Five",'Six','Seven','Eight','Nine']
queueLock = threading.Lock()
workQueue = Queue.Queue(9)
threads = []
threadID = 1

#create the queue
queueLock.acquire()
for word in nameList:
    workQueue.put(word)
queueLock.release()

#create threads
for tName in threadList:
    thread = myThread(threadID,tName,workQueue)
    threads.append(thread)
    threadID += 1

#start threads
for t in threads:
    t.start()

#wait queue be empty
while not workQueue.empty():
    pass

exitFlag = 1

#wait all thread stop
for t in threads:
    t.join()

print 'exiting main thread'
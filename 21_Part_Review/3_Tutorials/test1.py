import queue
queue1 = queue.Queue(maxsize = 3)
queue2 = queue.Queue(maxsize=4)

def safe_put(num, target):
    if target.full():
        target.get()
    target.put(num)

def show_queue(target):
    for i in range(target.qsize()):
        temp = target.get()
        print(temp)
        target.put(temp)

safe_put(1, queue1)
print()
print(queue1.qsize())
print()
safe_put(2, queue1)
safe_put(3, queue1)
safe_put(4, queue1)

for i in range(queue1.qsize()):
    print(queue1.get())

safe_put(5, queue2)
safe_put(6, queue2)
safe_put(7, queue2)
safe_put(8, queue2)
safe_put(9, queue2)

print()
show_queue(queue2)

print()
show_queue(queue2)
import time

time1 = time.time()
print time1

time.sleep(3)

print time.time() - time1


target_spans = [i + 1 for i in range(20)]

print target_spans
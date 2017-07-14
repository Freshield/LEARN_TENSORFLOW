import test3 as t3

t3.test()

log = ''

def test(log):
    log += 'lol'
    return log

log = test(log)
log = test(log)

print log


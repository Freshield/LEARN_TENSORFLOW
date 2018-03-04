import signal

class InputTimeoutError(Exception):
    pass

def interrupted(signum, frame):
    raise InputTimeoutError

def timer_input(time, words='Please input your name '):
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(time)

    try:
        name = raw_input(words+'in %s seconds:' % time)
    except InputTimeoutError:
        print('\ntimeout')
        name = 'None'

    signal.alarm(0)
    print('Your name is:%s' % name)
    return name

def wait_input(words='Please input number to choose:'):
    return raw_input(words)

str = timer_input(5)
print 'Received : ' + str
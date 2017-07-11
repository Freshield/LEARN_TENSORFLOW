import signal

class InputTimeoutError(Exception):
    pass

def interrupted(signum, frame):
    raise InputTimeoutError



def wait_input(time, words='Please input your name:'):
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(time)

    try:
        name = raw_input(words)
    except InputTimeoutError:
        print('\ntimeout')
        name = 'None'

    signal.alarm(0)
    print('Your name is:%s' % name)
    return name

str = wait_input(5)
print 'Received : ' + str
exit()
print 'here'

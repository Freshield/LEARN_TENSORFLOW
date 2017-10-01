class Log():
    def __init__(self):
        self.content = ''

    def add_content(self, content):
        self.content += content

    def clear_content(self):
        self.content = ''

log = Log()

print log.content

log.add_content('233')

print log.content

log.clear_content()

print log.content
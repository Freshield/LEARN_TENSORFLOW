class Employee:

    empCount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        print 'Total Emplyee %d' % Employee.empCount

    def displayEmployee(self):
        print 'Name: ', self.name, ', salary: ', self.salary

class Test:
    def prt(self):
        print(self)
        print(self.__class__)

emp1 = Employee('zara',2000)
emp2 = Employee('manni',5000)

emp1.displayEmployee()
emp2.displayEmployee()
print 'Total employee %d' % Employee.empCount

emp1.age = 7
print emp1.age
emp1.age = 8
print emp1.age
del emp1.age

print 'doc: ', Employee.__doc__
print 'name: ', Employee.__name__
print 'module: ', Employee.__module__
print 'bases: ', Employee.__bases__
print 'dict: ', Employee.__dict__
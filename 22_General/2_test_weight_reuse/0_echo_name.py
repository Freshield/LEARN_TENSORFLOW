#coding=utf-8

import os

##############################
#获得当前用户名，用来得到绝对路径
##############################
#--INPUT: 空
##############################
#--OUTPUT: name
#----name: String值，为当前用户名
##############################
def get_uname():
    name = os.popen('whoami')
    print(name.read())
    return name
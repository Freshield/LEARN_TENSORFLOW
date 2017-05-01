import file_filter as ff
import json
import os

rootdir = 'json'

dic_list = []

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        with open(os.path.join(parent, filename)) as jf:
            dic = json.load(jf)
            dic_list.append(dic)

print len(dic_list)
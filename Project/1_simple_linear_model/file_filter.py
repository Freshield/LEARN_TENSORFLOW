import os
import os.path
import json


################################################
#                                              #
# helper for store json file and load json file#
#                                              #
#                                              #
################################################


#to store the parameter into json file
def para_to_json(parent_dir, filename, target_dir):
    dic = {}
    with open(os.path.join(parent_dir, filename)) as f:
        for i in range(10):
            data = f.readline()
            if i > 1:
                line = data.strip()
                splited = line.split(' ')
                if splited[-1].find('-') != -1:
                    splited[-1] = splited[-1].split('-')[0]
                if i != 8:
                    dic[splited[0]] = float(splited[-1])
                else:
                    dic[splited[1]] = splited[-1]

        json_data = json.dumps(dic,sort_keys=True,indent =4,separators=(',', ': '))

        with open(target_dir+filename+'.json', 'w') as jf:
            jf.write(json_data)
        print 'done'

#load the json file and transfer it into dictionary
def json_to_dic(filename):
    with open(filename) as f:
        dic = json.load(f)
    return dic

#load lots json file into a list of dictionary
def json_to_dic_list(rootdir):
    dic_list = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            with open(os.path.join(parent, filename)) as jf:
                dic = json.load(jf)
                dic_list.append(dic)

    return dic_list


"""
rootdir = '/media/freshield/BUFFER/simple_linear/modules'

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        if filename.find('1.00000') != -1:
            print 'filename is', filename

            para_to_json(parent, filename, 'json/')

print 'finish'
"""
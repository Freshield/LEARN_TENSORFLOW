import json
import numpy as np
import basic_model as bm
#to save the dictionary to json file
#ver 1.0
def save_dic_to_json(dic, filename):
    with open(filename, 'w') as f:
        json.dump(dic, f)


#read loop_indexs from file
#ver 1.0
def read_loop_indexs(module_dir):
    filename = module_dir + 'loop_indexs'
    loop_indexs_dic = read_json_to_dic(filename)
    return np.array(loop_indexs_dic['loop_indexs'])


#to read json file and save to a dic
#ver 1.0
def read_json_to_dic(filename):
    with open(filename, 'r') as f:
        contents = json.load(f)
    return contents



a = np.arange(20)

print a

b = {'a' : a.tolist()}

print b

#save_dic_to_json(b, 'test.json')

c = read_loop_indexs('interrupt/')

print c

file_object = open('interrupt/interrupt')
try:
     all_the_text = file_object.read( )
finally:
     file_object.close( )

print all_the_text

print 'lol'

log = bm.Log()

print log.content

print 'lol'

log.add_content_from_file('interrupt/interrupt')

print log.content
import json

#the parameter need fill
#######################################################
#from network_model_example import *
SPAN=[10]
dir = '/home/freshield/Ciena_data/dataset_10k/model/'
epochs = 20
data_size = 10000
file_size = 1000
#how many loops do an evaluation
loop_eval_num = 5
#how many file do the valid
eval_last_num = 10
batch_size = 100
train_file_size = 800
valid_file_size = 100
test_file_size = 100
#hypers
reg = 0.000067
lr_rate = 0.002
lr_decay = 0.99
keep_prob_v = 0.9569
log_dir = 'logs/Link_CNN/'
module_dir = 'modules/Link_CNN/'
########################################################

para_whole_dataset_dic = {
    'SPAN' : SPAN,
    'dir' : dir,
    'epochs' : epochs,
    'data_size' : data_size,
    'file_size' : file_size,
    'loop_eval_num' : loop_eval_num,
    'batch_size' : batch_size,
    'train_file_size' : train_file_size,
    'valid_file_size' : valid_file_size,
    'test_file_size' : test_file_size,
    'reg' : reg,
    'lr_rate' : lr_rate,
    'lr_decay' : lr_decay,
    'keep_prob_v' : keep_prob_v,
    'log_dir' : log_dir,
    'module_dir' : module_dir,
    'eval_last_num' : eval_last_num
}

if __name__ == "__main__":
    print("one.py is being run directly")
    for (k, v) in para_whole_dataset_dic.items():
        print '%s : %s' % (k, v)

file_name = 'temp.json'

def save_dic_to_json(dic, filename):
    with open(filename, 'w') as f:
        json.dump(dic, f)

def read_json_to_dic(filename):
    with open(filename, 'r') as f:
        contents = json.load(f)
    return contents

def test():
    print 233

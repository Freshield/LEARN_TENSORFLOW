import os

from image_model import *

#################################################
#parameter you should refill
from train_model_example import *
#################################################

#wait the input and get the input number
#ver 1.0
def wait_input(words='Please input a number to choose:'):
    return raw_input(words)

#Return 'OK' for last layer flow
#ver 1.0
def return_OK():
    return 'OK'

#Return 'Back' to last layer
#ver 1.0
def return_Back():
    return 'Back'

#to control the main screen
#ver 1.0
def main_flow():
    print deep_fish_logo + main_screen
    while True:
        flow_number = wait_input()
        flow_number = int(flow_number)
        if flow_number < 1 or flow_number > 3:
            print "Error number, please re-input"
            continue
        else:
            #clear screen
            t = os.system('clear')
            back_value = main_flow_dic.get(flow_number)()
            if back_value == 'OK':
                break
            else:
                t = os.system('clear')
                print deep_fish_logo + main_screen
                continue


#to train a model
#ver 1.0
def train_flow():
    print deep_fish_logo + train_screen
    while True:
        flow_number = wait_input()
        flow_number = int(flow_number)
        if flow_number < 1 or flow_number > 4:
            print "Error number, please re-input"
            continue
        else:
            #clear screen
            t = os.system('clear')
            if flow_number == 1:
                back_value = train_flow_dic.get(flow_number)(para_whole_dataset_dic)
            elif flow_number == 2:
                back_value = train_flow_dic.get(flow_number)(para_whole_dataset_dic)
            elif flow_number == 3:
                back_value = train_flow_dic.get(flow_number)(para_whole_dataset_dic)
            else:
                back_value = train_flow_dic.get(flow_number)()
            if back_value == 'OK':
                return 'OK'
            elif back_value == 'Back':
                return 'Back'
            else:
                t = os.system('clear')
                print deep_fish_logo + train_screen
                continue

#show parameters
#ver 1.0
def show_parameters(para_dic):
    for (k, v) in para_dic.items():
        print '%s : %s' % (k, v)
    wait_input('Input c to continue:')
    return 'Done'

#change parameters
#ver 1.0
def change_parameters(para_dic):
    print 'Parameter here are:'
    for (k, v) in para_dic.items():
        print '%s : %s' % (k, v)
    print 'Which one do you want to change?'
    while True:
        para_name = wait_input('Input the parameter name or input e to go back:')
        if para_name == 'e':
            return 'Done'
        else:
            # check if have this parameter
            print para_name
            if not para_dic.has_key(para_name):
                print 'Do not have this parameter'
                continue
            else:
                #change the value
                para_value = wait_input('Input the parameter value:')
                #four types value
                if para_name == 'SPAN':
                    para_value = [int(para_value)]
                elif para_name == 'dir' or para_name == 'log_dir' or para_name == 'module_dir':
                    para_value = para_value
                elif para_name == 'reg' or para_name == 'lr_rate' or para_name == 'lr_decay' or para_name == 'keep_prob_v':
                    para_value = float(para_value)
                else:
                    para_value = int(para_value)

                #double check
                answer = wait_input('You sure want to change %s to %s?(y/n)' % (para_name, para_value))
                if answer == 'n':
                    continue
                else:
                    para_dic[para_name] = para_value
                    return 'Changed'




#to restore a model
#ver 1.0
def restore_flow():
    print deep_fish_logo + "restore flow"

#The main flow dictionary
#ver 1.0
main_flow_dic = {
    1: train_flow,
    2: restore_flow,
    3: exit
}

#The train flow dictionary
#ver 1.0
train_flow_dic = {
    1: show_parameters,
    2: change_parameters,
    3: train_whole_dataset_begin,
    4: return_Back
}

main_flow()
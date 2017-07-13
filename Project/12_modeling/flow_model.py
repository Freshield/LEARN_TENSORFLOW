import os

from image_model import *
from calculate_min_max_model import *
from recut_file_model import *
from train_model import *

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
        if flow_number < 1 or flow_number > 5:
            print "Error number, please re-input"
            continue
        else:
            #clear screen
            t = os.system('clear')
            back_value = main_flow_dic.get(flow_number)()
            if back_value == 'OK':
                break
            elif back_value == 'Back' or back_value == "Done":
                t = os.system('clear')
                print deep_fish_logo + main_screen
                continue
            else:
                t = os.system('clear')
                print deep_fish_logo + main_screen
                print back_value
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
                    return 'Changed the parameters'




#to restore a model
#ver 1.0
def restore_flow():
    print deep_fish_logo + "restore flow"

#to calculate the min and max value
#ver 1.0
def cal_min_max_flow():
    print deep_fish_logo
    while True:
        file_path = wait_input('Input the file path or input e to exit:')

        if file_path == 'e':
            return 'Back'
        elif os.path.exists(file_path) != True:
            print 'Error, file path do not exist'
            continue
        else:
            savename = wait_input('Input the savename or input e to exit:')

            if savename == 'e':
                return 'Back'
            else:
                datasize = wait_input('Input the datasize or input e to exit:')
                if datasize == 'e':
                    return 'Back'
                else:
                    datasize = int(datasize)
                    chunksize = wait_input('Input the chunksize or input e to exit:')
                    if chunksize == 'e':
                        return 'Back'
                    else:
                        chunksize = int(chunksize)
                        print ''
                        print 'The file path is %s' % file_path
                        print 'The savename is %s' % savename
                        print 'The datasize is %s' % datasize
                        print 'The chunksize is %s' % chunksize
                        answer = wait_input('Input y to start calcualte or input e to exit:')

                        if answer == 'e':
                            return 'Back'
                        else:
                            cal_min_max(file_path, savename, datasize, chunksize)

                            return 'Done and Save the min max at \n%s\n' % savename
    return 'Done'


#to norm and recut the file
#ver 1.0
def norm_recut_file_flow():
    print deep_fish_logo
    while True:
        filename = wait_input('Input the filename or input e to exit:')

        if filename == 'e':
            return 'Back'
        elif os.path.exists(filename) != True:
            print 'Error, file path do not exist'
            continue
        else:
            savePath = wait_input('Input the savePath or input e to exit:')

            if savePath == 'e':
                return 'Back'
            elif os.path.exists(savePath) != True:
                print 'Error, save path do not exist'
                continue
            else:
                minmax_name = wait_input('Input the minmax name or input e to exit:')

                if minmax_name == 'e':
                    return 'Back'
                elif os.path.exists(minmax_name) != True:
                    print 'Error, minmax path do not exist'
                    continue
                else:
                    datasize = wait_input('Input the datasize or input e to exit:')
                    if datasize == 'e':
                        return 'Back'
                    else:
                        datasize = int(datasize)
                        chunksize = wait_input('Input the chunksize or input e to exit:')
                        if chunksize == 'e':
                            return 'Back'
                        else:
                            chunksize = int(chunksize)
                            print ''
                            print 'The filename is %s' % filename
                            print 'The savePath is %s' % savePath
                            print 'The minmax name is %s' % minmax_name
                            print 'The datasize is %s' % datasize
                            print 'The chunksize is %s' % chunksize
                            answer = wait_input('Input y to start norm and recut or input e to exit:')

                            if answer == 'e':
                                return 'Back'
                            else:
                                norm_recut_dataset(filename,savePath,minmax_name,datasize,chunksize)

                                return 'Done and Save the files at \n%s\n' % savePath






#The main flow dictionary
#ver 1.0
main_flow_dic = {
    1: train_flow,
    2: restore_flow,
    3: cal_min_max_flow,
    4: norm_recut_file_flow,
    5: exit
}

#The train flow dictionary
#ver 1.0
train_flow_dic = {
    1: show_parameters,
    2: change_parameters,
    3: train_whole_dataset_begin,
    4: return_Back
}

import os

from image_model import *
from calculate_min_max_model import *
from recut_file_model import *
#from train_model import *
from restore_model import *
from transfer_learning import *

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
                wait_input('Input c to continue:')
            elif flow_number == 2:
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
                print back_value
                continue

#show parameters
#ver 1.0
def show_parameters(para_dic):
    for (k, v) in para_dic.items():
        print '%s : %s' % (k, v)
    return 'Done'

#change parameters flow
#ver 1.0
def change_parameters_flow(para_dic):
    print deep_fish_logo + change_para_screen
    while True:
        flow_number = wait_input()
        flow_number = int(flow_number)
        if flow_number < 1 or flow_number > 4:
            print "Error number, please re-input"
            continue
        else:
            # clear screen
            t = os.system('clear')
            #import from json file
            if flow_number == 1:
                back_value = imp_para_from_json(para_dic)
            #change by hand
            elif flow_number == 2:
                back_value = change_parameters(para_dic)
            else:
                back_value = 'Back'

            if back_value == 'OK':
                return 'OK'
            elif back_value == 'Back':
                return 'Done'
            else:
                t = os.system('clear')
                print deep_fish_logo + change_para_screen
                print back_value
                continue

#import parameters from json file
#ver 1.0
def imp_para_from_json(para_dic):
    while True:
        file_path = wait_input('Please input the json file path or input e to exit\nLike interrupt/parameters.json:')
        if file_path == 'e':
            return 'Done'
        elif os.path.exists(file_path) != True:
            print 'Error, file path do not exist'
            continue
        else:
            temp_dic = read_json_to_dic(file_path)
            show_parameters(temp_dic)
            answer = wait_input('\nParameters are here, You sure want to change?(y/n)')
            if answer == 'n':
                continue
            else:
                change_para_from_dic(para_dic, temp_dic)
                print 'Changed the parameters'
                print 'Your parameters now are:'
                show_parameters(para_dic)
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
                    try:
                        para_value = [int(para_value)]
                    except:
                        print "Error value"
                        continue
                elif para_name == 'dir' or para_name == 'log_dir' or para_name == 'module_dir':
                    try:
                        para_value = para_value
                    except:
                        print "Error value"
                        continue
                elif para_name == 'reg' or para_name == 'lr_rate' or para_name == 'lr_decay' or para_name == 'keep_prob_v':
                    try:
                        para_value = float(para_value)
                    except:
                        print "Error value"
                        continue
                else:
                    print para_value
                    try:
                        para_value = int(para_value)
                    except:
                        print "Error value"
                        continue

                #double check
                answer = wait_input('You sure want to change %s to %s?(y/n)' % (para_name, para_value))
                if answer == 'n':
                    continue
                else:
                    para_dic[para_name] = para_value
                    return 'Changed the parameters'

#train start flow
#ver 1.0
def train_start_flow():
    print deep_fish_logo + train_start_screen
    while True:
        flow_number = wait_input("Please input the Model number:")
        flow_number = int(flow_number)
        if flow_number < 1 or flow_number > 3:
            print "Error number, please re-input"
            continue
        else:
            # clear screen
            t = os.system('clear')
            back_value = model_dic[flow_number]
            if back_value == 'OK':
                return 'OK'
            elif back_value == 'Back':
                return 'Back'
            else:
                t = os.system('clear')
                print deep_fish_logo
                print "The model you choose is %s" % back_value
                show_parameters(para_whole_dataset_dic)
                print 'model name is %s' % back_value
                print
                while True:
                    answer = wait_input('Input y to start train or input e to go back:')
                    if answer == 'e':
                        return 'Back to train flow'
                    elif answer == 'y':
                        value = transfer_train_dataset_begin(para_whole_dataset_dic, back_value)
                        return value
                    else:
                        print 'Error input, please re-input'




#to restore a model
#ver 1.0
def restore_flow():
    print deep_fish_logo + restore_screen
    while True:
        flow_number = wait_input()
        flow_number = int(flow_number)
        if flow_number < 1 or flow_number > 4:
            print "Error number, please re-input"
            continue
        else:
            #clear screen
            t = os.system('clear')
            #show parameter
            if flow_number == 1:
                back_value = restore_flow_dic.get(flow_number)(para_whole_dataset_dic)
                wait_input('Input c to continue:')
            #change parameter
            elif flow_number == 2:
                back_value = restore_flow_dic.get(flow_number)(para_whole_dataset_dic)
            #restore begin
            else:
                back_value = restore_flow_dic.get(flow_number)()

            if back_value == 'OK':
                return 'OK'
            elif back_value == 'Back':
                return 'Back'
            else:
                t = os.system('clear')
                print deep_fish_logo + restore_screen
                print back_value
                continue



#restore start flow
#ver 1.0
def restore_start_flow():
    print deep_fish_logo + restore_start_screen
    while True:
        flow_number = wait_input("Please input the Model number:")
        flow_number = int(flow_number)
        if flow_number < 1 or flow_number > 3:
            print "Error number, please re-input"
            continue
        else:
            # clear screen
            t = os.system('clear')
            back_value = model_dic[flow_number]
            if back_value == 'OK':
                return 'OK'
            elif back_value == 'Back':
                return 'Back'
            else:
                t = os.system('clear')
                print deep_fish_logo
                print "The model you choose is %s" % back_value

                #ask for the log, model and loop indexs path
                while True:
                    print ''
                    model_path = wait_input('Please input the model path or input e to go back\nLike interrupt/module/module.ckpt:')
                    if model_path == 'e':
                        return 'Back to restore flow'
                    else:
                        while True:
                            print ''
                            log_path = wait_input(
                                'Please input the log path or input n to not use or input e to go back\nLike interrupt/interrupt:')
                            if log_path == 'e':
                                return 'Back to restore flow'
                            elif log_path != 'n' and os.path.exists(log_path) != True:
                                print 'Error, file path do not exist'
                                continue
                            else:
                                if log_path == 'n':
                                    log_path = None

                                while True:
                                    print ''
                                    loop_index_path = wait_input(
                                        'Please input the loop_index_path or input n to not use or input e to go back\nLike interrupt/loop_indexs:')
                                    if loop_index_path == 'e':
                                        return 'Back to restore flow'
                                    elif loop_index_path != 'n' and os.path.exists(loop_index_path) != True:
                                        print 'Error, file path do not exist'
                                        continue
                                    else:
                                        if loop_index_path == 'n':
                                            loop_index_path = None

                                        print ''
                                        show_parameters(para_whole_dataset_dic)
                                        print 'model name is %s' % back_value
                                        print 'model path is %s' % model_path
                                        print 'log path is %s' % log_path
                                        print 'loop indexs path is %s' % loop_index_path
                                        print
                                        while True:
                                            answer = wait_input('Input y to start restore and train or input e to go back:')
                                            if answer == 'e':
                                                return 'Back to restore flow'
                                            elif answer == 'y':
                                                value = restore_begin(para_whole_dataset_dic, back_value, model_path, log_path, loop_index_path)
                                                return value
                                            else:
                                                print 'Error input, please re-input'

#to calculate the min and max value
#ver 1.0
def cal_min_max_flow():
    print deep_fish_logo
    while True:
        file_path = wait_input('Input the file path or input e to exit\nLike /home/freshield/Ciena_data/dataset_10k/ciena10000.csv:')

        if file_path == 'e':
            return 'Back'
        elif os.path.exists(file_path) != True:
            print 'Error, file path do not exist'
            continue
        else:
            savename = wait_input('Input the savename or input e to exit\nLike /home/freshield/Ciena_data/dataset_10k/model/min_max_10k.csv:')

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
        filename = wait_input('Input the filename or input e to exit\nLike /home/freshield/Ciena_data/dataset_10k/ciena10000.csv:')

        if filename == 'e':
            return 'Back'
        elif os.path.exists(filename) != True:
            print 'Error, file path do not exist'
            continue
        else:
            savePath = wait_input('Input the savePath or input e to exit\nLike /home/freshield/Ciena_data/dataset_10k/model/:')

            if savePath == 'e':
                return 'Back'
            elif os.path.exists(savePath) != True:
                print 'Error, save path do not exist'
                continue
            else:
                minmax_name = wait_input('Input the minmax name or input e to exit\nLike /home/freshield/Ciena_data/dataset_10k/model/min_max_10k.csv:')

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

#The interrupt flow in training
#ver 1.0
def interrupt_flow(temp_para, sess, log, loop_indexs):
    print '\n'
    answer = timer_input(5)
    while True:
        if answer != 'i':
            break
        else:
            print '\n\n\n'
            print interrupt_screen

            temp_para_dic = store_para_to_dic(temp_para)
            flow_number = wait_input()
            flow_number = int(flow_number)
            #check parameters
            if flow_number == 1:
                show_parameters(temp_para_dic)
                continue
            #save model and exit
            elif flow_number == 2:
                path = wait_input('Please input the interrupt files you want to store\nLike interrupt/:')
                del_and_create_dir(path)
                save_dic_to_json(temp_para_dic, path + 'parameters.json')

                store_interrupt_module(path, sess, log, loop_indexs)
                store_interrupt_log(path, log)
                return 'Done'
            #continue
            elif flow_number == 3:
                return 'Continue'
            #change parameters
            elif flow_number == 4:
                print change_para_screen
                while True:
                    flow_number = wait_input()
                    flow_number = int(flow_number)
                    if flow_number < 1 or flow_number > 4:
                        print "Error number, please re-input"
                        continue
                    else:
                        # import from json file
                        if flow_number == 1:
                            back_value = imp_para_from_json(temp_para_dic)
                            data_para = get_para_from_dic(temp_para_dic)
                            change_para_from_array(temp_para, data_para)
                        # change by hand
                        elif flow_number == 2:
                            back_value = change_parameters(temp_para_dic)
                            data_para = get_para_from_dic(temp_para_dic)
                            change_para_from_array(temp_para, data_para)
                        #back
                        else:
                            back_value = 'Done'

                        if back_value == 'OK':
                            return 'OK'
                        elif back_value == 'Back':
                            return 'Back'
                        else:
                            print change_para_screen
                            print back_value
                            break
            #save model
            elif flow_number == 5:
                path = wait_input('Please input the interrupt files you want to store\nLike interrupt/:')
                del_and_create_dir(path)
                save_dic_to_json(temp_para_dic, path + 'parameters.json')

                store_interrupt_module(path, sess, log, loop_indexs)
                store_interrupt_log(path, log)
                continue
            else:
                print 'Error number, please re-input'
                continue





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
    2: change_parameters_flow,
    3: train_start_flow,
    4: return_Back
}

#The restore flow dictionary
#ver 1.0
restore_flow_dic = {
    1: show_parameters,
    2: change_parameters_flow,
    3: restore_start_flow,
    4: return_Back
}

#The model dic
#ver 1.0
model_dic = {
    1 : 'link_cnn',
    2 : 'resnet_link',
    3 : 'Back'
}


if __name__ == "__main__":
    main_flow()
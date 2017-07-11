import os

from image_model import *

#wait the input and get the input number
def wait_input(words='Please input a number to choose:'):
    return raw_input(words)


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
            main_flow_dic.get(flow_number)()


#to train a model
#ver 1.0
def train_flow():
    print deep_fish_logo + "train flow"

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

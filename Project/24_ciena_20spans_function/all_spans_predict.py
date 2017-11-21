from testing_model import *

for i in range(16):
    SPAN = [i + 4]
    testing_and_store(para_whole_dataset_dic, 'resnet_link', 'modules/span%d/module.ckpt' % SPAN[0],
                      '/media/freshield/DATA_U/', SPAN)
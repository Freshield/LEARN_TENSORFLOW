import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os

def get_files_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

def csv_to_tfrecord(filename, savename):

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    # create writer
    writer = tf.python_io.TFRecordWriter(savename)

    reader = pd.read_csv(filename, header=None, chunksize=1)

    count = 0

    for line in reader:
        print '    line %d begin convert' % count
        data = line.values.reshape((24081))

        real_C = data[0:12000]
        imag_C = data[12000:24000]
        netCD = data[24000:24001]
        length = data[24001:24021]
        power = data[24021:24041]
        ENLC = data[24041:24061]
        labels = data[24061:24081]

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'real_C': _float_feature(real_C),
                'imag_C': _float_feature(imag_C),
                'netCD' : _float_feature(netCD),
                '': _float_feature(length),
                'power' : _float_feature(power),
                'ENLC'  : _float_feature(ENLC),
                'labels': _float_feature(labels)
            }))
        writer.write(example.SerializeToString())
        count += 1

    writer.close()
    print 'one file successfully convert'

def convert_whole_dir_csv(dir_path, save_path):
    filename_list = get_files_name(dir_path)
    total_filenum = len(filename_list)
    count = 0
    for filename in filename_list:
        before_time = time.time()

        print 'file %d begin to convert' % count

        #look like Raw_data_xxx_train.csv
        the_name = filename.split('.')[0]
        savename = save_path + the_name + '.tfrecords'
        csv_to_tfrecord(dir_path+filename,savename)

        if count % 10 == 0 and count != 0:
            span_time = time.time() - before_time
            print "use %.2f second in 10 loop" % (span_time * 10)
            print "need %.2f minutes for all loop" % (((total_filenum - count) * span_time) / 60)

        # i += chunk.shape[0]
        count += 1
    print 'Done convert'




dir_path = '/media/freshield/New_2T_Data/Ciena_new_data/20spans/norm/'
save_path = '/media/freshield/New_2T_Data/Ciena_new_data/20spans/tfrecords_norm/'

convert_whole_dir_csv(dir_path,save_path)


""""""
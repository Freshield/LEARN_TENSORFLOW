import tensorflow as tf
import pandas as pd
import numpy as np
import Queue
import threading
import time
import os

def get_files_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

def csv_to_tfrecord(filename, savename, chunkSize):

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    # create writer
    writer = tf.python_io.TFRecordWriter(savename)

    reader = pd.read_csv(filename, header=None, chunksize=chunkSize)

    for chunk in reader:

        #count = 0

        temp_Data = chunk.values
        for lineNum in range(temp_Data.shape[0]):
            # print '    line %d begin convert' % count
            #if count % 50 == 0:
                #print '%s done %d lines convert' % (filename.split('/')[-1], count)
            data = temp_Data[lineNum]

            real_C = data[0:12000]
            imag_C = data[12000:24000]
            netCD = data[24000:24001]
            length = data[24001:24021]
            power = data[24021:24041]
            ENLC = data[24041:24061]
            labels = data[24061:24081]

            example = tf.train.Example(features=tf.train.Features(
                feature={'real_C': _float_feature(real_C), 'imag_C': _float_feature(imag_C),
                    'netCD': _float_feature(netCD), '': _float_feature(length), 'power': _float_feature(power),
                    'ENLC': _float_feature(ENLC), 'labels': _float_feature(labels)}))
            writer.write(example.SerializeToString())
            #count += 1

    writer.close()
    print '%s done convert' % filename.split('/')[-1]

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

        if count % 5 == 0 and count != 0:
            span_time = time.time() - before_time
            print "use %.2f second in 10 loop" % (span_time * 5)
            print "need %.2f minutes for all loop" % (((total_filenum - count) * span_time) / 60)

        # i += chunk.shape[0]
        count += 1
    print 'Done convert'

def multi_threads_convert_whole_dir(threadNums, dir_path, save_path, chunkSize):

    #create filenum class
    class Filenum():
        def __init__(self,total_num):
            self.total_num = total_num
            self.last_num = total_num

    #create thread class
    class myThread(threading.Thread):
        def __init__(self, threadID, name, q):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.name = name
            self.q = q

        def run(self):
            print 'starting ' + self.name
            process_data(self.name, self.q)
            print 'exiting ' + self.name

    # how to process the data
    def process_data(threadName, q):
        while not exitFlag:
            # get the lock
            queueLock.acquire()
            # check the queue status
            if not workQueue.empty():
                filename = q.get()
                queueLock.release()
                print '%s processing %s, %s' % (threadName, filename, time.ctime())
                # look like Raw_data_xxx_train.csv
                the_name = filename.split('.')[0]
                savename = save_path + the_name + '.tfrecords'
                csv_to_tfrecord(dir_path + filename, savename, chunkSize)
                #decay the file num
                filenumLock.acquire()
                filenum.total_num -= 1
                filenumLock.release()
            else:
                queueLock.release()
                time.sleep(5)

    filename_list = get_files_name(dir_path)
    filenum = Filenum(len(filename_list))

    #show when should exit
    exitFlag = 0

    # init
    threadList = ['Thread-%d' % (i + 1) for i in range(threadNums)]
    queueLock = threading.Lock()
    filenumLock = threading.Lock()
    workQueue = Queue.Queue(filenum.total_num)
    threads = []
    threadID = 1

    # create the queue
    queueLock.acquire()
    for word in filename_list:
        workQueue.put(word)
    queueLock.release()

    # create threads
    for tName in threadList:
        thread = myThread(threadID, tName, workQueue)
        threads.append(thread)
        threadID += 1

    # start threads
    for t in threads:
        t.start()

    # wait queue be empty
    while not workQueue.empty():
        before_time = time.time()
        time.sleep(300)
        span_time = time.time() - before_time
        filenumLock.acquire()
        span_num = filenum.last_num - filenum.total_num
        filenum.last_num = filenum.total_num
        print
        print 'rest %d files' % filenum.total_num
        print "last %.2f minutes process %d file" % (span_time / 60, span_num)
        print "need %.2f hours for all files" % (((filenum.total_num / span_num) * span_time) / 3600)
        print
        filenumLock.release()

    exitFlag = 1

    # wait all thread stop
    for t in threads:
        t.join()

    print 'exiting main thread'

##############################################################################3

dir_path = '/media/freshield/New_2T_Data/Ciena_new_data/20spans/norm/'
save_path = '/media/freshield/New_2T_Data/Ciena_new_data/20spans/tfrecords_norm/'

multi_threads_convert_whole_dir(8,dir_path,save_path,100)


""""""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import numpy as np
import h5py

class Weight_Bag:

    restore_dic = {}
    save_dic = {}

    def get_v_tensor(self,name):
        tensor = tf.Variable(self.restore_dic[name],name=name)
        return tensor

    def get_c_tensor(self,name):
        tensor = tf.constant(self.restore_dic[name],name=name)
        return tensor

    def save_tensor(self,sess,tensor,name):
        self.save_dic[name] = sess.run(tensor)

    def save_dic_to_hdf5(self,name):
        with h5py.File(name,'w') as f:
            for key,value in self.save_dic.items():
                f.create_dataset(key,data=value,compression='gzip')

    def hdf5_to_restore_dic(self,name):
        with h5py.File(name,'r') as f:
            for key in f.keys():
                self.restore_dic[key] = f[key].value

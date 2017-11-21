import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import os
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

file_dir = '/media/freshield/COASAIR1/CIENA/Result/logs/all_20spans_predict_data/'

files = file_name(file_dir)

files = sorted(files, key=lambda d : int(d.split('pan')[-1].split('_')[0]))
print files

file_name_dic = {}
for i in range(20):
    file_name_dic[i+1] = files[i]

def get_one_hot(data, depth):
    a = tf.placeholder(dtype=tf.int32,shape=(None))

    b = tf.one_hot(a,depth=depth)

    with tf.Session() as sess:
        feed_dict = {a: data}
        result = sess.run(b,feed_dict=feed_dict)
        return result

file_dir = '/media/freshield/COASAIR1/CIENA/Result/logs/all_20spans_predict_data/'
#file_name = 'span1_result_acc_0.9948.csv'
#file_name = 'span2_result_acc_0.9755.csv'
#file_name = 'span3_result_acc_0.9555.csv'
file_name = file_name_dic[20]
print file_name

data = pd.read_csv(file_dir+file_name).values

temp = data[:,0].astype(np.int32)
y_score = get_one_hot(temp,6)
temp = data[:,1].astype(np.int32)
y_test = get_one_hot(temp, 6)

n_classes = 6

#compute ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#plot
plt.figure()
#linewidth
lw = 2

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','blue','green','red','cyan','magenta','yellow'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc='lower right')
plt.show()
#one_hot_data = dpm.num_to_one_hot(predict_data,6)
#print one_hot_data.shape

#y_score = pd.read_csv(score_filename, header=None).values
#y_test = pd.read_csv(label_filename, header=None, dtype=np.int32).values

#print y_score.shape
#print y_test.shape

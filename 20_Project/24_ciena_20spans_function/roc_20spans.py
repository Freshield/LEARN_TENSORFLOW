import os
import pandas as pd
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return files

prob_file_dir = '/media/freshield/COASAIR1/CIENA/Result/logs/ciena_20spans_predict_prob/probs/'
lable_file_dir = '/media/freshield/COASAIR1/CIENA/Result/logs/ciena_20spans_predict_prob/lables/'

lable_files = file_name(lable_file_dir)
files = sorted(lable_files, key=lambda d : int(d.split('pan')[-1].split('_')[0]))
lable_file_dic = {}
for i in range(20):
    lable_file_dic[i+1] = files[i]

prob_files = file_name(prob_file_dir)
files = sorted(prob_files, key=lambda d : int(d.split('pan')[-1].split('_')[0]))
prob_file_dic = {}
for i in range(20):
    prob_file_dic[i+1] = files[i]

span_num = 20

score_filename = prob_file_dic[span_num]
label_filename = lable_file_dic[span_num]

y_score = pd.read_csv(prob_file_dir+score_filename, header=None).values
y_test = pd.read_csv(lable_file_dir+label_filename, header=None, dtype=np.int32).values

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
plt.title('SPAN %d ROC graph' % span_num)
plt.legend(loc='lower right')
plt.show()
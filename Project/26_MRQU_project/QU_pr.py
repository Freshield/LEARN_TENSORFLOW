import pandas as pd
import numpy as np

from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from itertools import cycle


score_filename = 'y_pred_value.csv'
label_filename = 'y_label_value.csv'

y_score = pd.read_csv(score_filename, header=None).values
Y_test = pd.read_csv(label_filename, header=None, dtype=np.int32).values

n_classes = 2


#for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:,i],y_score[:,i])

#plot pr curve for each class
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','blue','green','red','cyan','magenta','yellow'])

plt.figure(figsize=(7,8))
f_scores = np.linspace(0.2,0.8,num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y>=0], y[y>=0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9,y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('precision-recall for class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('extension of precision-recall curve to multi-class')
plt.legend(lines,labels,loc=(0,-.38),prop=dict(size=14))

plt.show()
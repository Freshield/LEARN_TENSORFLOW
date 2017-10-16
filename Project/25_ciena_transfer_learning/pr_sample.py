from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from itertools import cycle

#X is 150,4
#y is 150, total 3 class

iris = datasets.load_iris()
X = iris.data
y = iris.target

#add noisy
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

"""
#X is 150, 804 now

X_train, X_test, y_train, y_test = train_test_split(X[y<2], y[y<2],
                                                    test_size=.5,
                                                    random_state=random_state)

#X_train is 50,804
#X_test is 50,804
#y only have the class0 and class1

classifier = svm.LinearSVC(random_state=random_state)
classifier.fit(X_train,y_train)
y_score = classifier.decision_function(X_test)

average_precision = average_precision_score(y_test, y_score)

#begin to plot
precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')

plt.fill_between(recall,precision, step='post',alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0,1.05])
plt.xlim([0.0,1.0])
plt.title('2-class precision-recall curve: AUC={0:0.2f}'.format(average_precision))
plt.show()
"""

#multi-label data
#one hot
Y = label_binarize(y, classes=[0,1,2])
n_classes = Y.shape[1]

#X_train is 75,804
#X_test is 75,804

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=random_state)

classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)

#for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:,i],y_score[:,i])

precision['micro'], recall['micro'], _ = precision_recall_curve(Y_test.ravel(),
                                                                y_score.ravel())
average_precision['micro'] = average_precision_score(Y_test, y_score,
                                                     average='micro')

"""
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b',alpha=0.2,
         where='post')
plt.fill_between(recall['micro'],precision['micro'],step='post',alpha=0.2,
                 color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0,1.05])
plt.xlim([0.0,1.0])
plt.title(
    'average p recision score, micro-averaged over all classes: AUC={0:0.2f}'.format(average_precision['micro'])
)
plt.show()
"""

#plot pr curve for each class
colors = cycle(['navy','turquoise','darkorange','cornflowerblue','teal'])

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
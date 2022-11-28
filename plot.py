# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
print(matplotlib.get_cachedir())
plt.rcParams['font.sans-serif'] = ['Times New Roman']

labels = ['B', 'I', 'O']

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}

#test = pd.read_table('test_result/test_result.txt', header=None, names=['token','label', 'bio_only', 'upos', 'prediction']).dropna()
test = pd.read_table('val_result/bert_focal.txt', header=None, names=['token','label', 'bio_only', 'upos', 'prediction']).dropna()

y_true =[]
y_pred = []

for num in range(len(test['token'])):
    y_true.append(test['bio_only'][num])
    y_pred.append(test['prediction'][num])
y_pred.pop(0)
y_true.pop(0)
#print(y_true)
tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap='GnBu')
    plt.title(title)
    plt.colorbar().ax.tick_params(labelsize=30)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels,fontsize=30)
    plt.yticks(xlocations, labels,fontsize=30)
    plt.ylabel('Actual label',font1)
    plt.xlabel('Predicted label', font1)

cm = confusion_matrix(y_true, y_pred)
print(cm)
np.set_printoptions(precision=2)
 
cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis])
print (cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.005:
        plt.text(x_val, y_val, "%0.2f" % (c,) ,color='black', fontsize=30, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title=None)
# show confusion matrix
plt.savefig('test_result/confusion_matrix_val.jpg', format='jpg')
plt.show()


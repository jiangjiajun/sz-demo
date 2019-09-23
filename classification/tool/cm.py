'''compute confusion matrix
labels.txt: contain label name.
predict.txt: predict_label true_label
'''
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
#load labels.
labels = []
file = open('labels.txt', 'r')
lines = file.readlines()
for line in lines:
	labels.append(line.strip())
file.close()

y_true = []
y_pred = []
#load true and predict labels.
file = open('predict.txt', 'r')
lines = file.readlines()
for line in lines:
	y_true.append(int(line.split(" ")[1].strip()))
	y_pred.append(int(line.split(" ")[0].strip()))
file.close()

tick_marks = np.array(range(len(labels))) + 0.5
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap = plt.cm.autumn):#summer):#viridis):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.text(0, 2, "%0.2s" %(title,), color='red', fontsize=14, va='center', ha='center')
    plt.title('Predict label', y=1.04)
    #fig,ax=plt.subplots(1,1)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Confusion Matrix', fontsize=18)

cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_n = cm.astype('float')
plt.figure(figsize=(12,8), dpi=120)
#set the fontsize of label.
#for label in plt.gca().xaxis.get_ticklabels():
#    label.set_fontsize(8)
#text portion
ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_n[y_val][x_val]
    if (c > 0.01):
        print(x_val)
        plt.text(x_val, y_val, "%0.2d" %(c,), color='red', fontsize=14, va='center', ha='center')
#offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().yaxis.set_ticks_position('none')
plt.gca().invert_yaxis()
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_n, title='Normalized confusion matrix')
#show confusion matrix
plt.show()

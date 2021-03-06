import numpy as np
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, target_names, save_path, title='ConfusionMatrix.png',  
                                    cmap=None, color_bar=False, normalize=True):
    font = {'family':'sans-serif', 'weight':'medium', 'size':15}
    plt.rc('font', **font)

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    if color_bar:
        plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]*100),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(save_path + "/" + title)
    plt.show()
    

def precision_recall(cm):
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)

    # Specificity or true negative rate
    TNR = TN/(TN+FP) 

    # Precision or positive predictive value
    PPV = TP/(TP+FP)

    # Negative predictive value
    NPV = TN/(TN+FN)

    # Fall out or false positive rate
    FPR = FP/(FP+TN)

    # False negative rate
    FNR = FN/(TP+FN)

    # False discovery rate
    FDR = FP/(TP+FP)

    precision = TP / (TP+FP)  # ?????????

    recall = TP / (TP+FN)  # ?????????
    cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    WA = (np.diag(cm1)).sum()/4
    #WA = precision.sum()/4
    UA = TP.sum()/(cm.sum(axis=0)).sum()
    return WA, UA



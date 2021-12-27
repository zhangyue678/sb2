import numpy as np
from sklearn.metrics import confusion_matrix
from analysis import plot_confusion_matrix, precision_recall
import csv 
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


def result(save_folder, output_folder, plot_confusion=True):
    result = {}
    LABELS = []
    with open(save_folder+'/label_encoder.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        for i in range(4):
            if str(i)==line[9] and 'start' not in line:
                result[i] = line[:5]
    for i in range(4):
        LABELS.append(eval(result[i]))
    print(LABELS) 
   
    path = output_folder + '/predictions.csv'
    #path = '/data2/home/zhangyue/speechbrain/recipes/IEMOCAP/emotion_recognition/results/ECAPA-TDNN/1968/predictions.csv'
    with open(path, newline="") as csvfile:
        pred = []
        label = []
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        for row in reader:
            try:
                id = row["id"]
            except:
                del row["id"]
            pred.append(row['prediction'])
            label.append(row['true_value'])
    

    test_cm = confusion_matrix(label, pred)

    WA, UA = precision_recall(test_cm)
    res = [round(WA,4), round(UA,4)]
    
    with open(output_folder+'/train_log.txt', 'a+') as f:
        f.write('WA:{}, UA:{}'.format(res[0], res[1])+'\n')
    
    print('WA:{}, UA:{}'.format(res[0], res[1]))
    if plot_confusion:
        plot_confusion_matrix(test_cm, save_path=output_folder, normalize=True,
                                        target_names=LABELS, color_bar=True, title="ConfusionMatrix")

if __name__ == "__main__":

    hparams_file = 'hparams/train_ECAPA_LSTM.yaml'
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, '')

    result(hparams["save_folder"], hparams["output_folder"], plot_confusion=True)

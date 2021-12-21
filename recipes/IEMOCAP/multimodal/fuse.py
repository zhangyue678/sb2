import os
import csv


NLP_Path = '/data2/home/zhangyue/nlp/result/Bert/1969/nlp_predictions.csv'
Speech_CRNN_Path = '/data2/home/zhangyue/speechbrain/recipes/IEMOCAP/emotion_recognition/results/CRNN/1969/predictions.csv'
Speech_CRNN_Augment_Path = '/data2/home/zhangyue/speechbrain/recipes/IEMOCAP/emotion_recognition/results/CRNN_augment/1969/predictions.csv'

# id,prediction,true_value,outputs0,outputs1,outputs2,outputs3

with open(NLP_Path, newline="") as csvfile:
    nlp_dic = {}
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    for row in reader:
        d = {}
        id = row['id']
        d['prediction'] = row['prediction']
        d['true_value'] = row['true_value']
        d['outputs0'] = row['outputs0']
        d['outputs1'] = row['outputs1']
        d['outputs2'] = row['outputs2']
        d['outputs3'] = row['outputs3']
        nlp_dic[id] = d

with open(Speech_CRNN_Path, newline="") as csvfile:
    speech_CRNN_dic = {}
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    for row in reader:
        d = {}
        id = row['id']
        d['prediction'] = row['prediction']
        d['true_value'] = row['true_value']
        d['outputs0'] = row['outputs0']
        d['outputs1'] = row['outputs1']
        d['outputs2'] = row['outputs2']
        d['outputs3'] = row['outputs3']
        speech_CRNN_dic[id] = d

with open(Speech_CRNN_Augment_Path, newline="") as csvfile:
    speech_CRNN_Augment_dic = {}
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    for row in reader:
        d = {}
        id = row['id']
        d['prediction'] = row['prediction']
        d['true_value'] = row['true_value']
        d['outputs0'] = row['outputs0']
        d['outputs1'] = row['outputs1']
        d['outputs2'] = row['outputs2']
        d['outputs3'] = row['outputs3']
        speech_CRNN_Augment_dic[id] = d 


def weight_fuse(w1, w2, speech_dic=speech_CRNN_Augment_dic):
    true_values = []
    predictions = []
    for key,value in speech_dic.items():
        try:
            speech_dic[key]['outputs0'] and nlp_dic[key]['outputs0']
        
            true_values.append(int(speech_dic[key]['true_value']))
        
            d0 = w1 * float(speech_dic[key]['outputs0']) + w2 * float(nlp_dic[key]['outputs0'])
            d1 = w1 * float(speech_dic[key]['outputs1']) + w2 * float(nlp_dic[key]['outputs1'])
            d2 = w1 * float(speech_dic[key]['outputs2']) + w2 * float(nlp_dic[key]['outputs2'])
            d3 = w1 * float(speech_dic[key]['outputs3']) + w2 * float(nlp_dic[key]['outputs3'])
        
            res = [d0,d1,d2,d3]
        
            predictions.append(res.index(max(res)))
        except:
            continue
    return true_values, predictions

import numpy as np
from sklearn.metrics import confusion_matrix
from analysis import plot_confusion_matrix, precision_recall

LABELS = ["neu", "hap", "ang", "sad"]
def result(label, pred, output_folder, title="ConfusionMatrix"):

    test_cm = confusion_matrix(label, pred)

    WA, UA = precision_recall(test_cm)
    res = [round(WA,4), round(UA,4)]

    plot_confusion_matrix(test_cm, save_path=output_folder, normalize=True,
                                        target_names=LABELS, color_bar=True, title=title)

    return WA, UA
if __name__ == "__main__":
    fp = open('train_log.txt', 'a')
    fp.write('-'*36 + '\n')
    for i in range(0,11):
        w1 = 0 if i==0 else i/10
        w2 = round(1-w1,1)
        true_values, predictions = weight_fuse(w1, w2, speech_dic=speech_CRNN_Augment_dic)
        WA, UA = result(true_values, predictions, output_folder='./output_folder', title=None)
        true_values, predictions = weight_fuse(w1, 1-w1)
        WA, UA = round(WA, 2), round(UA, 2) 
        print("{}*speech + {}*nlp : WA:{}, UA:{}".format(w1, w2, WA, UA))
        fp.write("{}*speech + {}*nlp : WA:{}, UA:{}\n".format(w1, w2, WA, UA))
    fp.close()

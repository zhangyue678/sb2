import os
import shutil

dir = '/data2/home/zhangyue/speechbrain/recipes/IEMOCAP/emotion_recognition/results/ResNet/1969/save/'

if os.path.exists(dir):
    if not os.path.exists(dir+'models'):
        os.makedirs(dir+'models')
    for file in os.listdir(dir):
        if 'CKPT' in file:
            shutil.move(dir+file, dir+'models')

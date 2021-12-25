#!/bin/bash

echo "run the CUDA_LAUNCH_BLOCKING=1 python train.py hparams/train_ECAPA_LSTM.yaml";
count=0; while (( $count < 2 )); 
do  python moveModels.py && CUDA_LAUNCH_BLOCKING=1 python train.py hparams/train_ECAPA_LSTM.yaml && python moveModels.py;
((count=$count+1));
done;

echo "run the python train.py hparams/train.yaml";
count=0; while (( $count < 4 ));
do  python moveModels2.py && python train.py hparams/train.yaml && python moveModels2.py;
((count=$count+1));
done;

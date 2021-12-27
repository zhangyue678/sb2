#!/bin/bash


echo "run the python train_CNN.py hparams/train_ResNet.yaml";
count=0; while (( $count < 10 ));
do  python moveModels.py && python train_CNN.py hparams/train_ResNet.yaml && python moveModels.py;
((count=$count+1));
done;



<< EOF
echo "run the python train.py hparams/train.yaml";
count=0; while (( $count < 10 ));
do  python moveModels2.py && python train.py hparams/train.yaml && python moveModels2.py;
((count=$count+1));
done;


echo "run the CUDA_LAUNCH_BLOCKING=1 python train.py hparams/train_ECAPA_LSTM.yaml";
count=0; while (( $count < 5 ));
do  python moveModels.py && CUDA_LAUNCH_BLOCKING=1 python train.py hparams/train_ECAPA_LSTM.yaml && python moveModels.py;
((count=$count+1));
done;
EOF

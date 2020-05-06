#export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python train_efficientdet.py --network efficientdet-d1 --data-shape 640 --batch-size 16 \
--dataset coco --dataset-root ~/.mxnet/datasets --val-interval 10 --num-workers 8 --gpus 0,1,2,3 \
--lr 0.01 --lr-decay 0.1 --lr-decay-epoch 60,80 --warmup-epochs 2 \
--epochs 100 --syncbn --save-prefix ./checkpoint/
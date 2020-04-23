python train_efficientdet.py --network efficientdet-d1 --data-shape 640 --batch-size 8 \
--dataset coco --dataset-root /home/sw/.mxnet/datasets --val-interval 10 --num-workers 8 --gpus 1,2 \
--lr 0.001 --lr-decay 0.1 --lr-decay-epoch 60,80 --epochs 100 --syncbn --save-prefix ./checkpoint/
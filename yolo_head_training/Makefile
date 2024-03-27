tensorboard-daemon:
	nohup tensorboard --logdir=./checkpoints --host 0.0.0.0 --port 7777 > /dev/null 2>&1 &

all: yolo_3dheads_m yolo_3dheads_m_mixup yolo_3dheads_m_mosaic

yolo_3dheads_m:
	python train.py --config-name=yolo_3dheads_m \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/train \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/val \
    num_gpus=4 multi_gpu=DDP

yolo_3dheads_m_mixup:
	python train.py --config-name=yolo_3dheads_m \
    dataset_params=vgg_head_mixup \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/train \
    dataset_params.train_dataloader_params.batch_size=16 \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/val \
    num_gpus=4 multi_gpu=DDP

yolo_3dheads_m_mosaic:
	python train.py --config-name=yolo_3dheads_m \
    dataset_params=vgg_head_mosaic \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/train \
    dataset_params.train_dataloader_params.batch_size=16 \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/val \
    num_gpus=4 multi_gpu=DDP

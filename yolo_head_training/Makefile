tensorboard-daemon:
	nohup tensorboard --logdir=./checkpoints --host 0.0.0.0 --port 8888 > /dev/null 2>&1 &

yolo_3dheads_m:
	python train.py --config-name=yolo_heads_m \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/train \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/val \
    num_gpus=4 multi_gpu=DDP batch_size=24

yolo_3dheads_m_large:
	python train.py --config-name=yolo_heads_m \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/train \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/val \
    num_gpus=4 multi_gpu=DDP batch_size=24

yolo_heads_m_small_eugene:
	python train.py --config-name=yolo_heads_m_small_eugene num_gpus=4 multi_gpu=DDP

yolo_heads_m_extra_strides_small_eugene:
	python train.py --config-name=yolo_heads_m_extra_strides_small_eugene num_gpus=4 multi_gpu=DDP

yolo_heads_m_large_eugene:
	python train.py --config-name=yolo_heads_m_large_eugene \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/large \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/large \
    num_gpus=4 multi_gpu=DDP


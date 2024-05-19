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


yolo_heads_m_small_albu_eugene:
	python train.py --config-name=yolo_heads_m_small_eugene dataset_params=vgg_head_small_albu_augs \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP

yolo_heads_m_small_albu_eugene_crop_box:
	python train.py --config-name=yolo_heads_m_small_eugene dataset_params=vgg_head_small_albu_augs_crop_box \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP

yolo_heads_m_small_medium_albu_augs:
	python train.py --config-name=yolo_heads_m_small_eugene dataset_params=vgg_head_small_medium_albu_augs \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP


yolo_heads_m_large_medium_albu_augs_eugene:
	python train.py --config-name=yolo_heads_m_large_medium_albu_augs_eugene \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/large \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/large \
    num_gpus=4 multi_gpu=DDP


yolo_heads_l_small_eugene:
	python train.py --config-name=yolo_heads_l_small_eugene \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP

yolo_heads_m_small_eugene_grid_search:
	python train.py --config-name=yolo_heads_m_small_eugene \
    training_hyperparams=vgg_head_train_params \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP
	python train.py --config-name=yolo_heads_m_small_eugene \
    training_hyperparams=only_2d_keypoints \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP

yolo_heads_l_large_eugene:
	python train.py --config-name=yolo_heads_l_large_eugene \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/large \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/large \
    num_gpus=4 multi_gpu=DDP
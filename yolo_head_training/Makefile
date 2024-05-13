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

yolo_heads_m_small_eugene_grid_search:
	python train.py --config-name=yolo_heads_m_small_eugene \
    training_hyperparams.criterion_params.indexes_subset=0.5 experiment_suffix='sparse_05' \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP
	python train.py --config-name=yolo_heads_m_small_eugene \
    training_hyperparams.criterion_params.indexes_subset=0.4 experiment_suffix='sparse_04' \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP
	python train.py --config-name=yolo_heads_m_small_eugene \
    training_hyperparams.criterion_params.indexes_subset=0.3 experiment_suffix='sparse_03' \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP
	python train.py --config-name=yolo_heads_m_small_eugene \
    training_hyperparams.criterion_params.indexes_subset=0.2 experiment_suffix='sparse_02' \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP
	python train.py --config-name=yolo_heads_m_small_eugene \
    training_hyperparams.criterion_params.indexes_subset=0.1 experiment_suffix='sparse_01' \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP
	python train.py --config-name=yolo_heads_m_small_eugene \
    training_hyperparams.criterion_params.indexes_subset='445' experiment_suffix='sparse_default' \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/VGGHead/small \
    num_gpus=4 multi_gpu=DDP



tensorboard-daemon:
	nohup tensorboard --logdir=./checkpoints --host 0.0.0.0 --port 8888 > /dev/null 2>&1 &

yolo_heads_m_grid_search:
	python train.py --config-name=yolo_heads_m \
    training_hyperparams=vgg_head_train_params \
    dataset_params.train_dataset_params.data_dir=./VGGHead/small \
    dataset_params.val_dataset_params.data_dir=./VGGHead/small \
    num_gpus=4 multi_gpu=DDP

	python train.py --config-name=yolo_heads_m \
    training_hyperparams=only_2d_keypoints \
    dataset_params.train_dataset_params.data_dir=./VGGHead/small \
    dataset_params.val_dataset_params.data_dir=./VGGHead/small \
    num_gpus=4 multi_gpu=DDP

	python train.py --config-name=yolo_heads_m \
    training_hyperparams=only_2d_and_3d_keypoints \
    dataset_params.train_dataset_params.data_dir=./VGGHead/small \
    dataset_params.val_dataset_params.data_dir=./VGGHead/small \
    num_gpus=4 multi_gpu=DDP

yolo_heads_l:
	torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py --config-name=yolo_heads_l \
    dataset_params.train_dataset_params.data_dir=./VGGHead/large \
    dataset_params.val_dataset_params.data_dir=./VGGHead/large \
    num_gpus=4 multi_gpu=DDP

CUDA_VISIBLE_DEVICES=0 python trainval_WOUAF.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-base \
    --dataset_name HuggingFaceM4/COCO \
    --dataset_config_name 2014_captions --caption_column sentences_raw \
    --center_crop --random_flip \
    --dataloader_num_workers 4 \
    --train_steps_per_epoch 1_000 \
    --max_train_steps 50_000 \
    --phi_dimension 32 \
    --num_result 2 \
    --num_train_data 1000 \
    --num_val_data 400 \
    --resolution 256

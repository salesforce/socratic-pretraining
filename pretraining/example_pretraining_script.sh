CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_ENTITY="" WANDB_PROJECT="" \
TOKENIZERS_PARALLELISM=true \
python pretraining.py \
        --model_name_or_path facebook/bart-large \
        --do_train \
        --do_eval \
        --train_file "" \
        --validation_file "" \
        --output_dir ""\
        --overwrite_output_dir \
        --text_column src \
        --summary_column tgt \
        --cache_dir "" \
        --max_source_length 512 \
        --max_target_length 256 \
        --val_max_target_length 256 \
        --learning_rate 0.00003 \
        --optim adamw_hf \
        --lr_scheduler_type polynomial \
        --warmup_steps 5000 \
        --max_steps 100000 \
        --logging_steps 500 \
        --save_steps 10000 \
        --evaluation_strategy steps \
        --logging_strategy steps \
        --weight_decay 0.01 \
        --max_grad_norm 0.1 \
        --gradient_accumulation_steps 5 \
        --bf16 \
        --per_device_train_batch_size=12 \
        --per_device_eval_batch_size=12 \
        --report_to "wandb" \
        --dataloader_num_workers 2 \
        --load_best_model_at_end yes \


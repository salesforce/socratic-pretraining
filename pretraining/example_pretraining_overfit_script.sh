CUDA_VISIBLE_DEVICES=0,1 \
TOKENIZERS_PARALLELISM=true \
python pretraining.py \
        --model_name_or_path facebook/bart-large\
        --do_train \
        --do_eval \
        --train_file "" \
        --validation_file "" \
        --output_dir ""\
        --overwrite_output_dir \
        --text_column src \
        --summary_column labels \
        --max_source_length 512 \
        --max_target_length 256 \
        --val_max_target_length 256 \
        --learning_rate 0.00004 \
        --optim adamw_hf \
        --lr_scheduler_type polynomial \
        --warmup_steps 20 \
        --max_steps 1000 \
        --weight_decay 0.01 \
        --max_grad_norm 0.1 \
        --gradient_accumulation_steps 1 \
        --bf16 \
        --per_device_train_batch_size=12 \
        --per_device_eval_batch_size=24 \
        --overwrite_cache \
        --max_train_samples 100 \
        --max_eval_samples 100 \
        --logging_steps 50 \
        --eval_steps 50 \
        --save_strategy no \
        --evaluation_strategy steps \
        --logging_strategy steps \
        --report_to none\



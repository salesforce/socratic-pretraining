DATA=
OUTPUT_DIR=
PRED_PATH=$OUTPUT_DIR/test.predictions
TARGET_PATH=/test_original.target
MODEL_PATH=
WANDB_PROJECT_NAME=
PATH_TO_SUMEVAL=
GPU=6

# Training
WANDB_ENTITY=project WANDB_PROJECT=$WANDB_PROJECT_NAME CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --do_train \
  --train_file $DATA/train.jsonl \
  --do_eval \
  --validation_file $DATA/val.jsonl \
  --model_name_or_path $MODEL_PATH \
  --multiencoder_type bart \
  --multiencoder_max_num_chunks 32 \
  --multiencoder_stride \
  --max_source_len 512 \
  --learning_rate 0.000005 \
  --save_strategy epoch \
  --num_train_epochs 10 \
  --save_total_limit 1 \
  --gradient_checkpointing \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size 1 \
  --generation_max_len 512 \
  --val_max_target_length 512 \
  --evaluation_strategy epoch \
  --per_device_eval_batch_size 1 \
  --metric_for_best_model eval_mean_rouge \
  --compute_rouge_for_train \
  --predict_with_generate \
  --mode_token "<ask&answer>" \
  --logging_strategy epoch \
  --load_best_model_at_end \
  --report_to wandb \
  --bf16 \
  --seed 1 ;



## Inference
WANDB_ENTITY=project WANDB_PROJECT=$WANDB_PROJECT_NAME CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --do_predict \
  --test_file $DATA/test.jsonl \
  --model_name_or_path $OUTPUT_DIR \
  --multiencoder_type bart \
  --multiencoder_max_num_chunks 32 \
  --multiencoder_stride \
  --max_source_len 512 \
  --output_dir $OUTPUT_DIR \
  --generation_max_len 512 \
  --val_max_target_length 512 \
  --per_device_eval_batch_size 1 \
  --predict_with_generate \
  --prediction_path $PRED_PATH \
  --mode_token "<ask&answer>" \
  --seed 1 \
  --report_to wandb ;

## ROUGE
export ROUGE_HOME=$PATH_TO_SUMEVAL/SummEval/evaluation/summ_eval/ROUGE-1.5.5
pip uninstall pyrouge -y
pip install -U  git+https://github.com/bheinzerling/pyrouge.git
python qmsum_rouge.py --ref-path $TARGET_PATH --pred-paths $PRED_PATH
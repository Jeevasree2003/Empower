export CUDA_VISIBLE_DEVICES=0

# Run from EMPOWER-MODEL/. Writes a new prefix so output/best_pt1 is not overwritten.
python finetune.py \
    --max_source_length 384 \
    --max_knowledge_length 256 \
    --max_target_length 128 \
    --eval_max_gen_length 128 \
    --eval_min_gen_length 2 \
    --no_repeat_ngram_size 3 \
    --model_name_or_path "gpt2" \
    --data_dir "../data/preprocessed_v2" \
    --output_dir "output_kdpt_v2" \
    --learning_rate 3e-5 \
    --lr_scheduler "linear" \
    --warmup_steps 500 \
    --num_train_epochs 3 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --tuning_mode "pt1" \
    --eval_type "" \
    --mid_dim 800 \
    --preseqlen 16 \
    --val_metric "f1" \
    --save_top_k 3 \
    --early_stopping_patience 1 \
    --num_workers 0 \
    --overwrite_output_dir \
    --do_train

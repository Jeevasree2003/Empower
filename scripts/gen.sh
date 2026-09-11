export CUDA_VISIBLE_DEVICES=0

# Run from EMPOWER-MODEL/ so relative paths match kdpt.sh / rdpt.sh.
python finetune.py \
    --max_source_length 128 \
    --max_knowledge_length 256 \
    --max_target_length 128 \
    --eval_max_gen_length 128 \
    --eval_min_gen_length 20 \
    --eval_beams 5 \
    --no_repeat_ngram_size 3 \
    --resumed_ckpt_file "output_rdpt/loss=1.2422-step_count=0.ckpt" \
    --model_name_or_path "gpt2" \
    --pfxKlgModel_name_or_path "output/best_pt1" \
    --data_dir "../data/preprocessed" \
    --output_dir "output_gen_v1" \
    --eval_batch_size 8 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --tuning_mode "pt2" \
    --eval_type "" \
    --mid_dim 800 \
    --preseqlen 16

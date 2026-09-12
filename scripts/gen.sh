export CUDA_VISIBLE_DEVICES=0

# Run from EMPOWER-MODEL/. Point --resumed_ckpt_file at the best F1 RDPT ckpt
# after training (name is f1=...-step_count=....ckpt).
python finetune.py \
    --max_source_length 128 \
    --max_knowledge_length 256 \
    --max_target_length 128 \
    --eval_max_gen_length 128 \
    --eval_min_gen_length 2 \
    --eval_beams 1 \
    --no_repeat_ngram_size 3 \
    --resumed_ckpt_file "output_rdpt_v3/best.ckpt" \
    --model_name_or_path "gpt2" \
    --pfxKlgModel_name_or_path "output_kdpt_v2/best_pt1" \
    --data_dir "../data/preprocessed_v2" \
    --output_dir "output_gen_v2" \
    --eval_batch_size 8 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --tuning_mode "pt2" \
    --eval_type "" \
    --mid_dim 800 \
    --preseqlen 64 \
    --klg_preseqlen 16

export CUDA_VISIBLE_DEVICES=1
MODEL= "gpt2"
###DATASET="wow or cmu_dog"
python finetune.py \
    --max_source_length 128 \
    --max_knowledge_length 256 \
    --max_target_length 128 \
    --eval_max_gen_length 128 \
    --resumed_ckpt_file ".ckpt file trained on RDPT phase" \
    --model_name_or_path "gpt2" \
    --pfxKlgModel_name_or_path "Model trained on KDPT phase" \
    --data_dir "/dataset" \
    --output_dir " " \
    --eval_batch_size 8 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --tuning_mode "pt2" \
    --eval_type "seen" \
    --mid_dim 800 \
    --preseqlen 64
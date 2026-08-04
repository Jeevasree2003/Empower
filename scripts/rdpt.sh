export CUDA_VISIBLE_DEVICES=0
MODEL="gpt2"
##DATASET="wow or cmu_dog"
python finetune.py \
    --max_source_length 128 \
    --max_knowledge_length 256 \
    --max_target_length 128 \
    --eval_max_gen_length 128 \
    --model_name_or_path "gpt2" \
    --pfxKlgModel_name_or_path "Model trained on KDPT phase(/path/best_pt1)" \
    --data_dir "dataset" \
    --output_dir " " \
    --learning_rate 3e-5 \
    --lr_scheduler "linear" \
    --warmup_steps 2000 \
    --num_train_epochs 30 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --tuning_mode "pt2" \
    --eval_type "seen" \
    --mid_dim 800 \
    --preseqlen 64 \
    --do_train
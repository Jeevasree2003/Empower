export CUDA_VISIBLE_DEVICES=2
MODEL="gpt2"
##DATASET="wow"
python  finetune.py \
	--max_source_length 384 \
	--max_knowledge_length 256 \
	--max_target_length 128 \
	--eval_max_gen_length 128 \
	--model_name_or_path "gpt2" \
	--data_dir "../data/preprocessed"  \
	--output_dir "output" \
	--learning_rate 3e-5 \
	--lr_scheduler "linear" \
	--warmup_steps 2000 \
	--num_train_epochs 10 \
	--train_batch_size 8 \
	--eval_batch_size 8\
    --gradient_accumulation_steps 1 \
    --n_train -1 \
	--n_val -1 \
	--n_test -1 \
	--tuning_mode "pt1" \
	--eval_type "" \
	--mid_dim 800 \
	--preseqlen 16 \
	--do_train
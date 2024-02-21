# normal textsum training
python 1_main_textsum_with_trainer.py

# textsum training with accelerate
accelerate config --config_file textsum_config.yaml
accelerate launch --config_file textsum_config.yaml 2_main_textsum_with_accelerate.py

# big model inference with accelerate
python 3_accelerate_big_transformers_inference.py

# textsum training with deepspeed
accelerate config --config_file "deepspeed_config.yaml"
accelerate launch --config_file deepspeed_config.yaml 4_main_textsum_with_accelerate_deepspeed.py

# basic Causal LM training with run_clm_no_trainer.py
python 5_run_clm_no_trainer.py \
	--dataset_name wikitext \
	--dataset_config_name wikitext-2-raw-v1 \
	--model_name_or_path gpt2 \
	--output_dir output/test-clm \
	--per_device_train_batch_size 1 \
	--max_train_steps 5 \
	--block_size 32

# run Megatron-LM training with accelerate
accelerate config --config_file "megatron_gpt_config.yaml"
accelerate launch --config_file megatron_gpt_config.yaml \
	6_megatron_lm_gpt_pretraining.py \
	--dataset_name wikitext \
	--dataset_config_name wikitext-2-raw-v1 \
	--model_name_or_path gpt2 \
	--output_dir output/test-clm-megatron \
	--per_device_train_batch_size 1 \
	--max_train_steps 5 \
	--block_size 32

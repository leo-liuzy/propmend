export CUDA_VISIBLE_DEVICES=1

train_set_size=40_000 # I just use large number to include all the data

python -m run +alg=mend +experiment=controlled_ripple_edit +model=qwen2.5-1.5B-qa-sft-qa-additional-max val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K

# PropMEND with RippleEdit
# python -m run +alg=mend +experiment=ripple_edit +model=qwen2.5-1.5B-qa-sft-qa-additional-max val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K
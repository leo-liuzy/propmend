export CUDA_VISIBLE_DEVICES=1

train_set_size=40_000 # I just use large number to include all the data

python -m run +alg=mend +experiment=syn_story +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-max val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K

# PropMEND with RippleEdit
# python -m run +alg=mend +experiment=ripple_edits +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-max val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K
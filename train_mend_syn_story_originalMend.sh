export CUDA_VISIBLE_DEVICES=1

# standard MEND config
python -m run +alg=mend +experiment=zsre +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-top val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 mend.shared=True data.rephrase=True

# MEND with midupper to match PropMEND(Mid-Upper)
# python -m run +alg=mend +experiment=syn_story_mend +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-midupper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 mend.shared=True train_prefix=4K
export CUDA_VISIBLE_DEVICES=1

train_set_size=100_000 # I just use 100k to include all the data


# paraphrase (in outer loop)
python -m run +alg=mend +experiment=controlled_ripple_edit_ablate_propagation +model=qwen2.5-1.5B-qa-sft-qa-additional val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K

# cpt (in inner loop)
# python -m run +alg=mend +experiment=controlled_ripple_edit_ablate_cpt +model=llama3.2-1B-qa-sft-qa-additional-midupper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K

# Top layers
# python -m run +alg=mend +experiment=controlled_ripple_edit +model=qwen2.5-1.5B-qa-sft-qa-additional-top val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K
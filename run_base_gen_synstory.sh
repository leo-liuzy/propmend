export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    # [exp_dir_name]=run_id
    # e.g., [llama3.2-1B_on_zsre-full]=2025-02-10_08-19-14_2641409766
    
)



n_val=100
prompt=no
exp_dir_name="ripple_edits_all_heavy-noshare-mid-upper3"
archive=${name2id[$exp_dir_name]}

base_model_name=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10

ice=True # prepend; for base, set to False.
test_data=4K_test_id


python run_base_generate_synstory.py +alg=mend eval_only=True generation.save_dir=synstory_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False +ice=${ice} +test_data=${test_data}

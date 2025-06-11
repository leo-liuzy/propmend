export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    # [exp_dir_name]=run_id
    # e.g., [llama3.2-1B_on_zsre-full]=2025-02-10_08-19-14_2641409766
    
)



n_val=100
prompt=no

base_model_name=llama3.2-1B-qa-sft-qa-additional

ice=True # prepend; for base, set to False.
test_data=4K_test_id


python run_base_generate_controlled_ripple.py +alg=mend eval_only=True generation.save_dir=controlled_ripple_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False +ice=${ice} +test_data=${test_data}

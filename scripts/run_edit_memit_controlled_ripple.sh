export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    # [exp_dir_name]=run_id
    # e.g., [llama3.2-1B_on_zsre-full]=2025-02-10_08-19-14_2641409766
    
)


n_val=200 # 50
prompt=no
task=controlled_ripple_edit
mom2_dataset="controlled_ripple_4K"

for config_name in llama3.2-1B-qa-sft-qa-additional-estimated # -wiki
do

for test_data in "4K_test_id" 
do

python run_edit_memit_controlled_ripple.py +alg=mend +experiment=${task} +model=llama3.2-1B-qa-sft-qa-additional-midupper eval_only=True generation.save_dir=controlled_ripple_exp_output/${config_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +test_data=${test_data} +config_name=${config_name} +mom2_dataset=${mom2_dataset}

done
done
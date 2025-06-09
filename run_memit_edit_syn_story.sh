export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    # [exp_dir_name]=run_id
    # e.g., [llama3.2-1B_on_zsre-full]=2025-02-10_08-19-14_2641409766
    
)


n_val=200 # 50
prompt=no
task=syn_story
mom2_dataset="synstory_4K"

for config_name in llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-estimated # -wiki
do

for test_data in "4K_test_id" 
do

python run_memit_edit_syn_story.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-midupper3 eval_only=True generation.save_dir=synstory_exp_output/${config_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +test_data=${test_data} +config_name=${config_name} +mom2_dataset=${mom2_dataset}

done
done
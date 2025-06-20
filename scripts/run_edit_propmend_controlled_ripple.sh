export CUDA_VISIBLE_DEVICES=2

declare -A name2id=(
    # [exp_dir_name]=run_id
    # e.g., [llama3.2-1B_on_zsre-full]=2025-02-10_08-19-14_2641409766
    
)

n_val=500
prompt=no
task=controlled_ripple_edit

exp_dir_name=30K_heavy_share_midupper3
archive=${name2id[$exp_dir_name]}

for date_data in 4K_test_ood 4K_test_ood-entity 4K_test_ood-relation 4K_test_id
do

python run_edit_propmend_controlled_ripple.py +alg=mend +experiment=${task} +model=llama3.2-1B-qa-sft-qa-additional-midupper archive=${archive} eval_only=True generation.save_dir=controlled_ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +test_data=${test_data} mend.shared=False +exp_name=${exp_dir_name} 


done
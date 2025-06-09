export CUDA_VISIBLE_DEVICES=1

declare -A name2id=(
    # [exp_dir_name]=run_id
    # e.g., [llama3.2-1B_on_zsre-full]=2025-02-10_08-19-14_2641409766
    
)


n_val=200
prompt=no
task=ripple_edits

exp_dir_name=ripple_edits_all_original_mend_share_midupper
archive=${name2id[$exp_dir_name]}

for test_data in all
do

python run_original_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-midupper archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +test_data=${test_data} mend.shared=True


done

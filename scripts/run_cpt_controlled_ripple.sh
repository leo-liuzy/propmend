export CUDA_VISIBLE_DEVICES=0

gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
bs=1
per_device_train_batch_size=1
grad_acc=$((bs / gpu_count / per_device_train_batch_size))

weight_decay=0.1
max_grad_norm=1.0

seed=42
max_seq_length=1024
epoch=4
lr=1e-5
epoch=4


base_model_name="Qwen2.5-1.5B-qa-sft-qa-additional"

test_data=4K_test_id
text_data="text"

for tunable_params in all # "midupper3-mlp" # "midupper3-mlp" # "all" 
do 
for example_idx in {0..49} # {51..349}
do

echo "Test data: ${test_data}"
echo "Example idx: ${example_idx}"

python cpt_controlled_ripple.py \
    --seed=${seed} \
    --output_dir="${PWD}/models" \
    --learning_rate=${lr} \
    --lr_scheduler_type=constant \
    --weight_decay=${weight_decay} \
    --per_device_train_batch_size=${per_device_train_batch_size} \
    --gradient_accumulation_steps=${grad_acc} \
    --max_seq_length=${max_seq_length} \
    --max_grad_norm=${max_grad_norm} \
    --optim="adamw_torch" \
    --dataset_text_field="text" \
    --bf16=True \
    --eval_strategy="no" \
    --save_strategy="no" \
    --logging_strategy="steps" \
    --logging_first_step=True \
    --logging_steps=1 \
    --report_to="wandb" \
    --num_train_epochs=${epoch} \
    --run_name="propagator-clm-baseline" \
    --example_idx=${example_idx} \
    --report_to="none" \
    --spec_question=False \
    --test_data=${test_data} \
    --text_data=${text_data} \
    --tunable_params=${tunable_params} \
    --base_model_name=${base_model_name} 

done
done
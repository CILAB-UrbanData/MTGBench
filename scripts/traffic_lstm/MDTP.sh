export CUDA_VISIBLE_DEVICES=1

model_name=MDTP

python -u run.py \
  --task_name TrafficLSTM \
  --is_training 1 \
  --root_path ./data/NYC_Taxi\&Bike/MDTP/processed \
  --model $model_name \
  --data MDTP \
  --normalization True \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --lradj 'type4' \
  --itr 1 \
  --train_epochs 200 \
  --patience 20
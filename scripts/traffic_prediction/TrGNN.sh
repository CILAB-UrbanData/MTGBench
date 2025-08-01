export CUDA_VISIBLE_DEVICES=1

model_name=TrGNN

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path data/sf_data/TrGNN/ \
  --model $model_name \
  --data TrGNN \
  --normalization True \
  --learning_rate 0.0001 \
  --batch_size 2 \
  --lradj 'type5' \
  --itr 1 \
  --train_epochs 2 \
  --patience 20
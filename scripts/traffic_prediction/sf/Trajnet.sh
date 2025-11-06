export CUDA_VISIBLE_DEVICES=1

model_name=Trajnet

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path data/sf_data/Trajnet/processed \
  --model $model_name \
  --data Trajnet \
  --normalization True \
  --learning_rate 0.0001 \
  --batch_size 256 \
  --lradj 'type5' \
  --itr 1 \
  --train_epochs 100 \
  --patience 20
model_name=TrGNN

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path data/GaiyaData/TrGNN/processed \
  --model $model_name \
  --data DiDiTrGNN \
  --normalization True \
  --learning_rate 0.004 \
  --batch_size 32 \
  --lradj 'type5' \
  --itr 1 \
  --train_epochs 100 \
  --patience 20 \
  --NumofRoads 3345 \
  --start_date 20161101\
  --end_date 20161107 \
  --pre_steps 1
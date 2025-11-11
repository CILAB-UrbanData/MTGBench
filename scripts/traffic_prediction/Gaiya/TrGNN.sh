model_name=TrGNN

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 0 \
  --root_path data/GaiyaData/TrGNN/processed \
  --model $model_name \
  --data DiDiTrGNN \
  --normalization True \
  --learning_rate 0.002 \
  --batch_size 64 \
  --lr_scheduler 'cosine' \
  --learner 'adamw' \
  --adamw_beta1 0.9 \
  --adamw_beta2 0.95 \
  --adamw_weight_decay 0.001 \
  --grad_clip 1.0 \
  --itr 1 \
  --train_epochs 100 \
  --patience 20 \
  --NumofRoads 3555 \
  --start_date 20161101\
  --end_date 20161130 \
  --pre_steps 1 \
  --lr_istorch 
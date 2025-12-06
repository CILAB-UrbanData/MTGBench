model_name=TrGNN

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path data/GaiyaData/TrGNN/processed \
  --shp_file road_shp_with_extra.shp \
  --traj_file traj_converted.csv \
  --length_col 'length_m' \
  --model $model_name \
  --data chengdu \
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
  --min_flow_count 1 \
  --NumofRoads 3556 \
  --time_interval 10 \
  --start_date 20161101\
  --end_date 20161130 \
  --pre_steps 1 \
  --lr_istorch 
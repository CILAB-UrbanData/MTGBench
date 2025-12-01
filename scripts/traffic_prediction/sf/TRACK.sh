model_name=TRACK

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path data/sf_data/raw \
  --model $model_name \
  --data SF \
  --NumofRoads 7239 \
  --min_flow_count 100 \
  --feat_col 'length,lanes,oneway' \
  --static_feat_dim 5 \
  --traj_file traj_train_100.csv \
  --shp_file map/edges.shp \
  --time_interval 10 \
  --learning_rate 0.001 \
  --batch_size 4 \
  --lr_scheduler 'cosine' \
  --lr_istorch \
  --learner 'adamw' \
  --adamw_beta1 0.9 \
  --adamw_beta2 0.999 \
  --adamw_weight_decay 0.01 \
  --itr 1 \
  --train_epochs 50 \
  --patience 10 \
  --pre_steps 1 \
  --load_pretrained \
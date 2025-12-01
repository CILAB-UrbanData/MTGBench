model_name=TRACK

python -u run.py \
  --task_name TRACK_pretrain \
  --is_training 1 \
  --root_path data/Porto/match_jll \
  --model $model_name \
  --data porto \
  --NumofRoads 2977 \
  --learning_rate 0.001 \
  --batch_size 32 \
  --lr_scheduler 'cosine' \
  --lr_istorch \
  --learner 'adamw' \
  --adamw_beta1 0.9 \
  --adamw_beta2 0.999 \
  --adamw_weight_decay 0.1 \
  --itr 1 \
  --train_epochs 50 \
  --patience 10 \
  --pre_steps 1 \
  --min_flow_count 10000 \
  --time_interval 120 \
  --static_feat_dim 4 \
  --feat_col 'length,oneway' \
  --shp_file map/edges.shp \
  --traj_file traj_porto.csv \ 

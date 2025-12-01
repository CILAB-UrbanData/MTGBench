model_name=Trajnet

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path data/sf_data/raw \
  --shp_file map/edges.shp \
  --traj_file traj_train_100.csv \
  --model $model_name \
  --data Trajnet \
  --normalization True \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --lradj 'type5' \
  --itr 1 \
  --train_epochs 100 \
  --patience 20 \
  --NumofRoads 1717 \
  --time_interval 10 \
  --tstride 15 \
  --min_flow_count 1000 \

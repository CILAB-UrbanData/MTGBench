model_name=Trajnet

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path data/Porto/match_jll \
  --shp_file map/edges.shp \
  --model $model_name \
  --data porto \
  --normalization True \
  --tstride 300 \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --lradj 'type5' \
  --itr 1 \
  --train_epochs 100 \
  --patience 20 \
  --NumofRoads 1477 \
  --traj_file traj_porto.csv \
  --min_flow_count 25000\
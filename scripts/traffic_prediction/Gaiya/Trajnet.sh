model_name=Trajnet

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 0 \
  --root_path data/GaiyaData/TRACK \
  --shp_file roads_chengdu.shp \
  --traj_file traj_converted.csv \
  --model $model_name \
  --data chengdu \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --lradj 'type5' \
  --itr 1 \
  --train_epochs 100 \
  --patience 20 \
  --NumofRoads 1716 \
  --min_flow_count 2000\
  --time_interval 10 \
  --tstride 20 \
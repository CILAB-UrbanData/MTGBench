model_name=Trajnet

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path data/GaiyaData/TRACK \
  --shp_file roads_chengdu.shp \
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
  --traj_file traj_converted.csv \
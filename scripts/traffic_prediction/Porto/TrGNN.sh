model_name=TrGNN

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path data/Porto/match_jll \
  --model $model_name \
  --data TrGNN \
  --normalization True \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --lradj 'type5' \
  --itr 1 \
  --train_epochs 100 \
  --patience 20 \
  --pre_steps 1 \
  --NumofRoads 2977 \
  --traj_file traj_porto.csv \
  --preprocess_path preprocess_TrGNNporto.pkl \
  --time_interval 10000 \
export CUDA_VISIBLE_DEVICES=0,1

model_name=MDTPsingle

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path data/GaiyaData/MDTP \
  --mdtp_taxi_path merged_with_grid.csv \
  --model $model_name \
  --data chengdu \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --lradj 'type4' \
  --N_regions 202 \
  --itr 10 \
  --train_epochs 200 \
  --patience 20
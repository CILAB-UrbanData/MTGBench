model_name=MDTP

python -u run.py \
  --task_name TrafficPrediction\
  --is_training 0 \
  --root_path data/NYC/MDTP\
  --mdtp_taxi_path part.0.parquet \
  --mdtp_bike_path bike_data_with_region.csv \
  --model $model_name \
  --data NYCFULL \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --lradj 'type4' \
  --N_regions 264 \
  --itr 1 \
  --train_epochs 200 \
  --patience 20 \
  --dropout 0.5
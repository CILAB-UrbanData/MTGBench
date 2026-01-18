model_name=MDTPsingle

python -u run.py \
  --task_name TrafficPrediction\
  --is_training 1 \
  --root_path data/Porto/MDTP\
  --mdtp_taxi_path TrajwithRegion.csv \
  --model $model_name \
  --data porto \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --lradj 'type4' \
  --N_regions 195 \
  --itr 10 \
  --train_epochs 200 \
  --patience 20 \
  --dropout 0.5
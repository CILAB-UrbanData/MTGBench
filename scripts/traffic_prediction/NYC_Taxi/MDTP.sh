model_name=MDTP

python -u run.py \
  --task_name TrafficPrediction\
  --is_training 1 \
  --root_path ./data/NYC_Taxi\&Bike/MDTP/processed \
  --model $model_name \
  --data MDTP \
  --normalization True \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --lradj 'type4' \
  --N_regions 264 \
  --itr 10 \
  --train_epochs 200 \
  --patience 20 \
  --dropout 0.5
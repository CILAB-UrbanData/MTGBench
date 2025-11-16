export CUDA_VISIBLE_DEVICES=0,1

model_name=MDTPsingle

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 0 \
  --root_path /mnt/nas/home/cilab/wyx_ws/Traffic-Benchmark/data/GaiyaData/MDTP/processed \
  --model $model_name \
  --data OtherForMDTP \
  --normalization True \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --lradj 'type4' \
  --N_regions 202 \
  --itr 10 \
  --train_epochs 200 \
  --patience 20
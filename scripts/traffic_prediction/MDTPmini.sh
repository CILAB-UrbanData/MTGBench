export CUDA_VISIBLE_DEVICES=1

model_name=MDTPmini

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path /mnt/nas/home/cilab/wyx_ws/Traffic-Benchmark/data/GaiyaData/MDTP/processed \
  --model $model_name \
  --data GaiyaForMDTP \
  --normalization True \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --lradj 'type4' \
  --N_nodes 206 \
  --itr 1 \
  --train_epochs 200 \
  --patience 20
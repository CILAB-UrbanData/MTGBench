model_name=Trajnet

python -u run.py \
  --task_name TrafficPrediction \
  --is_training 1 \
  --root_path /mnt/nas/home/cilab/wyx_ws/Traffic-Benchmark/data/GaiyaData/Trajnet/processed \
  --model $model_name \
  --data DiDiTrajnet \
  --normalization True \
  --learning_rate 0.0001 \
  --batch_size 256 \
  --lradj 'type5' \
  --itr 1 \
  --train_epochs 100 \
  --patience 20 \
  --T3 0 \
  --n_s 3345 \
  --adj 'adj_sparse.pkl' \
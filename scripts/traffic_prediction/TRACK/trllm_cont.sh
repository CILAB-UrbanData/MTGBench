export CUDA_VISIBLE_DEVICES=1

model_name=TRACK.trllm_cont

python -u run.py \
  --task_name TRACK_trllm_cont \
  --is_training 1 \
  --root_path data/sf_data/Trajnet/processed \
  --model $model_name \
  --data TRACK_Gaiya \
  --normalization True \
  --learning_rate 0.0001 \
  --batch_size 256 \
  --lradj 'type5' \
  --itr 1 \
  --train_epochs 2 \
  --patience 20
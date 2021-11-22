python run.py \
  --dataset_path=data/preprocessed/ml-1m.txt \
  --hidden_units=50 \
  --num_blocks=2 \
  --num_heads=1 \
  --device=0 \
  --test=True\
  --model_path=output/SASRec_epoch_420.pth.tar
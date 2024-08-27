#python train_target_model.py --dataset citeseer_full --target-model gat  --num-hidden 256 --gpu -1

python3 attack.py --dataset citeseer_full --target-model-dim 256 --num-hidden 256 --target-model gat --surrogate-model gin --recovery-from prediction --query_ratio 1.0 --structure original --gpu -1

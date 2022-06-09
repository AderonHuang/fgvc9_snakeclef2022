EXP_NAME=run_simclr

nohup \
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
    --cfg ./configs/MetaFG_2_384.yaml  \
    --batch-size 10 \
    --tag $EXP_NAME \
    --lr 5e-5 \
    --min-lr 5e-7 \
    --warmup-lr 5e-8 \
    --epochs 300 \
    --warmup-epochs 20 \
    --dataset /path/to/data_label_unlabel/ \
    --pretrain /ckpt_epoch_299.pth \
    --accumulation-steps 2 \
    --opts DATA.IMG_SIZE 384 TRAIN.AUTO_RESUME False \
    --output output \
    --amp-opt-level O0 \
    --root /path/trainset \
    --nb-classes 1572 \
    --num-workers 8 \
    >> output/$EXP_NAME.out &

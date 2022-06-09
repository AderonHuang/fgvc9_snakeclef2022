EXP_NAME=run
nohup \
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
    --cfg ./configs/MetaFG.yaml \
    --batch-size 32 \
    --tag $EXP_NAME \
    --lr 5e-5 \
    --min-lr 5e-7 \
    --warmup-lr 5e-8 \
    --epochs 300 \
    --warmup-epochs 20 \
    --dataset imagenet \
    --pretrain \
    --opts DATA.IMG_SIZE 384 TRAIN.AUTO_RESUME False \
    --output output \
    --amp-opt-level O1 \
    --root  \
    --nb-classes 1572 \
    >> output/$EXP_NAME.out &

# fgvc9_snakeclef2022
## steps
train
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg ./configs/MetaFG.yaml --batch-size 32 --tag OUTPUT_TAG --lr 5e-5  --min-lr 5e-7 --warmup-lr 5e-8 --epochs 300 --warmup-epochs 20 --dataset imagenet --pretrain --opts DATA.IMG_SIZE 384 TRAIN.AUTO_RESUME False --output output  --amp-opt-level O1 --root --nb-classes 1572
```
test
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --eval --cfg ./configs/MetaFG.yaml --batch-size 10 --tag OUTPUT_TAG --dataset snakeclef2022test --resume $MODEL_PATH --opts DATA.IMG_SIZE 384 TRAIN.AUTO_RESUME False
```
post process and ensamble
After runing test, we will get result_snakeclef2022test.tc, use ``post_process.py`` which indicate the final output of a single model, we can ensamble the model outputs by runing ``fuse_logits.py``
## results
tesm | score
:----:|:-----:|
ARM(ours)|0.89436
base|0.89101
GG|0.85409

Our code are partly based on [metaformer](https://github.com/dqshuai/MetaFormer) and [moco](https://github.com/dqshuai/MetaFormer)

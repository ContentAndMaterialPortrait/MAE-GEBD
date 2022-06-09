
### linux path
~/code/cvpr2022-workshop/GEBD-MAEpro

### 1、pretrain
```bash
OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=8 run_main_pretraining3D.py
--data_path
/PATH/TO/cvpr2022/DATA/
--mask_ratio
0.25
--model
pretrain_mae_base_patch4_128
--batch_size
256
--opt
adamw
--opt_betas
0.9
0.95
--warmup_epochs
40
--epochs
1600
--output_dir
/PATH/output/
```

### 2、finetuning
```bash
OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=8 run_main_class_finetuning3D.py 
--data_path
/PATH/TO/cvpr2022/DATA/train.pkl 
--eval_data_path /PATH/TO/cvpr2022/DATA/val.pkl
--nb_classes 40 
--data_set image_GEBD3D 
--finetune /dockerdata/output/checkpoint.pth 
--output_dir /dockerdata/output/
--eval_f1_path ./data_split/k400_all.pkl 
--batch_size 6 
--opt adamw 
--opt_betas 0.9 0.999 
--weight_decay 0.05 
--epochs 1200 
--lr 0.00003
# Now, it only supports pretrained models with normalized pixel targets
```

### 3.env

check in  requirements.txt
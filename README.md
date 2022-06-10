# MAE-GEBD

This is the code for track 1 of competition LOVEU@CVPR2022.

We have two kinds of models. One of them is MAE-based models, the other is base model. Base models is 
inherited from [LOVEU-CVPR2021](https://github.com/hello-jinwoo/LOVEU-CVPR2021). And we applied a series improvement on it.

## PART1: MAE-based models
This MAE3D repository is built upon [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch), thanks very much!
```
./MAE3D/tools/prepare_data_MAE_2fps_img.py  # create 40 frame images
./MAE3D/tools/get_GEBD_class_data.py        # get pre-training / finetuning data
./MAE3D/run_main_pretraining3D.py           #  pre-training script
./MAE3D/run_main_class_finetuning3D.py      #  finetuning script
```
detail in ./MAE3D/README.md
## PART2: Base models

### Setup
Here, we provide our basic setup. 
- python 3.8
- torch 1.8.1
- numpy 1.19.2
- matplotlib 3.4.1
- tqdm 

### Video Features
You can downlaod our video feuatures [here](https://drive.google.com/drive/folders/1AJl177kLvl1YtaFBb9QmiUAQ5o5qsjq9?usp=sharing).

You may locate feature data in 'data' folder of this repository.

### Materials 
You can download models [here](https://drive.google.com/drive/folders/11OkI6SeRLd7Ewc9JyuoCUSHb29bu4foN).

### Implementation
You can change some values in config.py in both models. 

#### Train
For both models, you can train model just using below code.
```
python main.py
```

#### Validate
If you want to validate models using saved model, follow below.

- *using_similarity_map*
```
python validate.py --model $MODEL_NAME --fold $FOLD_NUM --sigma $SIGMA_VALUE
```
For example, 
```
python validate.py --model model_main_fold_0_s_0.4_8397.pt --fold 0 --sigma 0.4
```
And you will get the output f1:0.8397.

<hr>

- *sf_tsn_each_branch*
```
python validate.py --model_sf $MODEL_SF_NAME --model_tsn $MODEL_TSN_NAME --sigma $SIGMA_VALUE --fold $FOLD_NUM
```
For example, 
```
python validate.py --model_sf model_sf_simple_fold_0_s_-1_8384.pt --model_tsn model_tsn_simple_fold_0_s_-1_8384.pt --sigma -1 --fold 0
```
And you will get the output f1:0.8384

### Test with ensemble
We predict the result by ensembling models from different folds(0~9) and model architecture(*using_similarity*, *sf_tsn_each_branch*, *mae-based*).

We save a probability score for each model and use it to produce final prediction.

With the probability scores, you can predict the final event boundary following below code in ensemble folder. Put all your prob_results.pkl files in *ensemble/base_models*, and then:
```
python test.py --ver $VERSION_NAME_YOU_WANT
```

If you want ensemble MAE-GEBD models as well, put MAE-based models in *ensemble/mae_models*, and then:
```
python test_mae.py --ver $VERSION_NAME_YOU_WANT
```

Then, there will be the test result in results folder.



## SLR train
## $config_file -> eg: config_resnet18_0.9
## $pretrained_model -> eg: ASSETS_PATH/seg_weights/ocrnet.HRNet_industrious-chicken.pth
python3 train_slr.py --admm-train --sparsity-type irregular --config-file $config_file --snapshot $pretrained_model

## retrain
python3 train_slr.py --masked-retrain --sparsity-type irregular --config-file $config_file
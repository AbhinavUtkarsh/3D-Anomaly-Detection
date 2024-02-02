data_dir=./datasets/Real_AD_3D_multiview/
exp_name="Testrun"

# run with ResNet backbone
# python main.py --category  diamond --n-views 27 --no-fpfh False --data-path $data_dir --exp-name $exp_name --backbone resnet18

# run with NasNet large backbone
python main.py --category  diamond --n-views 27 --no-fpfh False --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
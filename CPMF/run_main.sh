data_dir=./datasets/Real_AD_3D_multi_view/
exp_name="Testrun"

# run with ResNet backbone
# python main.py --category  diamond --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone resnet18


# run with NasNet large backbone
python main.py --category  diamond --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
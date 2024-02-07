data_dir=./datasets/Real_AD_3D_multi_view/
exp_name="Testrun"

# run with ResNet backbone
# python main.py --category  diamond --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone resnet18


# run with NasNet large backbone
python main.py --category  airplane --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
python main.py --category  car --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
python main.py --category  chicken --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
python main.py --category  diamond --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
python main.py --category  duck --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
python main.py --category  gemstone --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
python main.py --category  shell --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
python main.py --category  toffee --n-views 27 --no-fpfh True --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
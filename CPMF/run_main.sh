data_dir=./datasets/Real_AD_3D_multiview/
exp_name="Testrun"

# run with ResNet backbone
# python main.py --category  diamond --n-views 27 --no-fpfh False --data-path $data_dir --exp-name $exp_name --backbone resnet18

# run with Pooling-based Vision Transformer (PiT) backbone
# python main.py --category  diamond --n-views 27 --no-fpfh False --data-path $data_dir --exp-name $exp_name --backbone pit_s_224

# run with Swin Transformer backbone
# python main.py --category  diamond --n-views 27 --no-fpfh False --data-path $data_dir --exp-name $exp_name --backbone swin_s3_small_224

# run with NasNet large backbone
python main.py --category  diamond --n-views 27 --no-fpfh False --data-path $data_dir --exp-name $exp_name --backbone nasnetalarge
data_dir=./datasets/Real_AD_3D_multiview/
exp_name="Testrun"


python train.py --category  diamond --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
# python train.py --category  airplane --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
# python train.py --category  candy --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
# python train.py --category  car --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
# python train.py --category  chicken --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
# python train.py --category  duck --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
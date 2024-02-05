data_dir=./datasets/Real_AD_3D_multi_view/
exp_name="Testrun"

#by default train.py will take all the classes in the multi-view directory and train seperate models on each class.

python train.py --category  diamond --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
python train.py --category  airplane --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
python train.py --category  candy --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
python train.py --category  car --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
python train.py --category  chicken --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
python train.py --category  duck --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
python train.py --category  shell --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
python train.py --category  toffee --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
python train.py --category  gemstone --dataset_path $data_dir --ckpt_path ./checkpoint_pointnet
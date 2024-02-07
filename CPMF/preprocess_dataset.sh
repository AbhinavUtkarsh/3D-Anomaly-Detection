data_dir=./../datasets/Real_AD_3D/
save_dir=./../datasets/Real_AD_3D_multi_view/
export DISPLAY=:0
cd utils

# remove the background (commented because diamond class already preprocessed)
python preprocessing.py --dataset_path $data_dir


python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category airplane --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category candybar --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category car --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category chicken --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category diamond --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category duck --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category gemstone --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category toffee --save-dir $save_dir
python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category shell --save-dir $save_dir

cd ..

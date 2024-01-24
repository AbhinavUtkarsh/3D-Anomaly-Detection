data_dir=./../datasets/Real_AD_3D/
save_dir=./../datasets/Real_AD_3D_multiview
export DISPLAY=:0
cd utils

# remove the background
python preprocessing.py --dataset_path $data_dir

# note: if you run on a cluster without screen, you may need to complie headless open3d:
# http://www.open3d.org/docs/release/tutorial/visualization/headless_rendering.html
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category airplane --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category candybar --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category car --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category chicken --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category diamond --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category duck --save-dir $save_dir


# CLASSES FROM OLD DATASET
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category bagel --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category cable_gland --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category carrot --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category cookie --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category dowel --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category foam --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category peach --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category potato --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category rope --save-dir $save_dir
#python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category tire --save-dir $save_dir
cd ..

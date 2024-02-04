import os
import glob
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils.utils import * 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.spatial import cKDTree

def sample_query_points_entire_pc(target_point_clouds):
    """
    Modify the function to work with the entire point cloud and add noise to the points.
    """
    # Ensure target_point_clouds is a numpy array, not a PyTorch tensor
    if isinstance(target_point_clouds, torch.Tensor):
        target_point_clouds = target_point_clouds.cpu().numpy()


    target_point_clouds = target_point_clouds.squeeze()
    sample = []
    pnts = target_point_clouds
    ptree = cKDTree(pnts)

    # Calculate sigmas for noise addition
    sigmas = []
    for p in np.array_split(pnts, 100, axis=0):
        d = ptree.query(p, 51)
        sigmas.append(d[0][:, -1])
    sigmas = np.concatenate(sigmas)

    # Add noise to the entire point cloud
    noisy_points = pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape)
    sample.append(noisy_points)

    sample = np.array(sample).reshape(-1, 3)

    return sample


def normal_points(ps_gt, translation=False): 
    tt =  0
    if((np.max(ps_gt[:,0])-np.min(ps_gt[:,0]))>(np.max(ps_gt[:,1])-np.min(ps_gt[:,1]))):
        tt = (np.max(ps_gt[:,0])-np.min(ps_gt[:,0]))
    else:
        tt = (np.max(ps_gt[:,1])-np.min(ps_gt[:,1]))
    if(tt < (np.max(ps_gt[:,2])-np.min(ps_gt[:,2]))):
        tt = (np.max(ps_gt[:,2])-np.min(ps_gt[:,2]))
     
    tt = 10/(10*tt)
    ps_gt = ps_gt*tt

    if(translation):
        t = np.mean(ps_gt,axis = 0)
        ps_gt = ps_gt - t

    return ps_gt, (t , tt)

def data_process(npz_dir_path):
    points_gt_all = []
    files_path = npz_dir_path
    if(os.path.exists(files_path)):
        load_data = np.load(files_path)
        points_gt, _ = normal_points(load_data, True)
        points_gt_all.append(points_gt)
    return points_gt_all


class RealAD3D(Dataset):
    def __init__(self, datasets_path):
        self.img_paths = datasets_path

class RealAD3DTrain(RealAD3D):
    def __init__(self,datasets_path):
        super().__init__(datasets_path=datasets_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        npz_path = self.img_paths[idx]
        #load npz data
        points_gt_all= data_process(npz_path)
        return points_gt_all
    
    

def get_train_data_loader(datasets_path):
    dataset = RealAD3DTrain(datasets_path=datasets_path)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    return data_loader
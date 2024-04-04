"""
This 3D backbone architecture draws significant inspiration and implementation from the 
Shape-Guided model by Yu-Min Chu, Chieh Liu, Ting-I Hsieh, Hwann-Tzong Chen, and Tyng-Luh Liu.
See their research here: https://openreview.net/pdf?id=IkSGn9fcPz
We encourage users to consult the original work: https://github.com/jayliu0313/Shape-Guided/blob/main/core/model.py 
"""

import numpy as np
import torch
from torch.autograd import Variable
from ptflops import get_model_complexity_info
import torch.nn.functional as F
import torch.nn as nn
from model.pointnet_util import *
from model.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


def pytorch_safe_norm(x, epsilon=1e-12, axis=None):
    return torch.sqrt(torch.sum(x ** 2, axis=axis) + epsilon)

class encoder_BN_2(nn.Module):
    def __init__(self,num_class = 128,normal_channel=False):
        super(encoder_BN, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        #self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        #self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
        self.bn3 = nn.BatchNorm1d(num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        return x

class encoder_BN(nn.Module):
    def __init__(self):
        super(encoder_BN, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x):
       
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)


        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        return x
    
## decoder with initial
class local_NIF(nn.Module):
    def __init__(self):
        super(local_NIF, self).__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(3, 512)
        self.fc3 = nn.Linear(512*2, 512)
        torch.nn.init.constant_(self.fc3.bias, 0.0)
        torch.nn.init.normal_(self.fc3.weight, 0.0, np.sqrt(2) / np.sqrt(512))

        for i in range(10): #change it to 10
            fc4 = nn.Linear(512, 512)
            torch.nn.init.constant_(fc4.bias, 0.0)
            torch.nn.init.normal_(fc4.weight, 0.0, np.sqrt(2) / np.sqrt(512))
            fc4 = nn.utils.weight_norm(fc4)
            setattr(self, "fc4" + str(i), fc4)

        self.fc5 = nn.Linear(512, 33)
        torch.nn.init.constant_(  self.fc5.bias, 0.0)
        torch.nn.init.normal_( self.fc5.weight, 0.0, np.sqrt(2) / np.sqrt(512))
        self.fc5 = nn.utils.weight_norm( self.fc5)
        
        self.fc6 = nn.Linear(33, 1)
        torch.nn.init.constant_(self.fc6.bias, -0.5)
        torch.nn.init.normal_(self.fc6.weight, mean=2*np.sqrt(np.pi) / np.sqrt(33), std=0.000001)
        #self.bn = nn.BatchNorm1d(512)

    def forward(self, points_feature, input_points):

        feature_f = F.relu(self.fc1(points_feature))
        net = F.relu(self.fc2(input_points))
        net = torch.concat([net, feature_f], dim=2)
        net = F.relu(self.fc3(net))
        for i in range(7):
            fc4 = getattr(self, "fc4" + str(i))
            net = F.relu(fc4(net))

        features = self.fc5(net)
        sdf = self.fc6(features)
        return features, sdf

    def get_gradient(self, points_feature, input_points):
        input_points.requires_grad = True
        features ,sdf = self.forward(points_feature, input_points)
        gradient = torch.autograd.grad(
            sdf,
            input_points,   
            torch.ones_like(sdf, requires_grad=False, device=sdf.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)
        normal_p_lenght = torch.unsqueeze(
            pytorch_safe_norm(gradient[0], axis=-1), -1)
        grad_norm = gradient[0] / normal_p_lenght
        g_point = input_points - sdf * grad_norm
        return features ,g_point


class SDF_Model(nn.Module):
    def __init__(self,point_net_backbone):
        super(SDF_Model, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if point_net_backbone == 'pointnet':
            self.encoder = encoder_BN()
        else:
            self.encoder = encoder_BN_2()
        
        self.NIF = local_NIF()

    def forward(self, gt,noisy_points):
        num_points = noisy_points.shape[1]
        gt = torch.permute(gt, (0, 2, 1))
        pointnet_feature = self.encoder(gt)
        pointnet_feature = pointnet_feature.unsqueeze(0)
        pointnet_feature = torch.mean(pointnet_feature, dim=1, keepdim=True)
        pointnet_feature = pointnet_feature.squeeze(1)
        point_feature = torch.tile(torch.unsqueeze(
            pointnet_feature, 1), [1, num_points, 1])
        features, g_point = self.NIF.get_gradient(point_feature, noisy_points)
        return features, g_point

    def predict(self, gt,noisy_points):
        num_points = noisy_points.shape[1]
        gt = torch.permute(gt, (0, 2, 1))
        pointnet_feature = self.encoder(gt)
        pointnet_feature = pointnet_feature.unsqueeze(0)
        pointnet_feature = torch.mean(pointnet_feature, dim=1, keepdim=True)
        pointnet_feature = pointnet_feature.squeeze(1)
        point_feature = torch.tile(torch.unsqueeze(
            pointnet_feature, 1), [1, num_points, 1])
        features, _ = self.NIF.forward(point_feature, noisy_points)
        return features
        
    def get_feature(self, point):
        point = torch.permute(point, (0, 2, 1))
        return self.encoder(point)

    def get_sdf(self, point_feature, input_points):
        return self.NIF(point_feature, input_points)

    def freeze_model(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        for p in self.NIF.parameters():
            p.requires_grad_(False)
            
def prepare_input(resolution):
    x1 = torch.FloatTensor(1, 500, 3)
    x2 = torch.FloatTensor(1, 500, 3)
    return dict(gt = x1, input_points = x2)
 
if __name__ == '__main__':
    gt = Variable(torch.rand(32, 500, 3))
    sampled_point = Variable(torch.rand(32, 500, 3),  requires_grad=True)
    sdf_model = SDF_Model(500)

    flops, params = get_model_complexity_info(sdf_model, (1, 500, 3), input_constructor=prepare_input, as_strings=True, print_per_layer_stat=True)
    print('Flops: ' + flops)
    print('Params: ' + params)

from torch import optim
import datetime
import glob
import argparse
from tqdm import tqdm

from utils.utils import *
from model.model import SDF_Model
from data.data import get_train_data_loader,sample_query_points_entire_pc
from model.pointnet_util import sample_and_group


def object_to_str(parameter):
    keys = list(parameter.keys())
    vals = list(parameter.values())
    index = np.argsort(keys)
    res = ""
    for i in index:
        if callable(vals[i]):
            v = vals[i].__name__
        else:
            v = str(vals[i])
        res += "%30s: %s\n" % (str(keys[i]), v)
    return res

# saving parameters in a pickle file and a txt file
def save(parameter, file_name):
    # Pickle the parameter object
    pickle_data(file_name + ".pickle", parameter)
    
    # Write the string representation to a text file
    with open(file_name + ".txt", 'w') as fout:
        fout.write(object_to_str(parameter))


class Training():
    def __init__(self, hyperparameters, class_, point_net_backbone="pointnet"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # fetching hyperparameters
        if hyperparameters is not None:
            self.epoch = hyperparameters["epoch"]
            learning_rate = hyperparameters["learning_rate"]
            self.checkpoint_path = hyperparameters["checkpoint_path"]
            self.current_iteration = 0
        self.curr_class_ = class_
        
        # Initializing pointnet2 model
        self.sdf_model = SDF_Model(point_net_backbone).to(self.device)
        if hyperparameters is not None:
            self.optimizer = optim.Adam(self.sdf_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        

    def train(self, train_dataloader):
        # Open the log file once and keep it open during training
        with open(osp.join(self.checkpoint_path, "train_states.txt"), "a", 1) as log_train_file:
            
            for epoch in range(0,self.epoch + 1):
                loss_tracker = []

                for points in tqdm(train_dataloader, desc=f"Currently Pre-training the #Epoch: {epoch}"):
                    points = points[0]

                    torch_points = points.to(torch.float32).to(self.device)
                    patches = sample_and_group(torch_points, npoint=64, nsample=500)
                    patches = patches.squeeze(0)
                    # First, ensure the tensor is moved to the CPU before converting it to a NumPy array
                    noisy_points = sample_query_points_entire_pc(torch_points.cpu().numpy())    
                    
                    # Then, you can convert the noisy_points back to a PyTorch tensor and move it to the CUDA device
                    noisy_points = torch.from_numpy(noisy_points).to(torch.float32).to(self.device)
                    
                    noisy_points = noisy_points.unsqueeze(0)
                    point_feature, g_point = self.sdf_model(patches,noisy_points)

                    loss = torch.linalg.norm((torch_points - g_point), ord=2, dim=-1).mean()

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Track and log the loss
                    loss_tracker.append(loss.detach().cpu().numpy())
                    
                log_train_file.write("#Epoch: %d, Loss: %f\n" % (epoch, loss.detach().cpu().numpy()))

                if (epoch % 25 == 0):
                    print('**** Saving the model **** #Epoch:', epoch, ' Loss:', loss.detach().cpu().numpy())
                    self.save_checkpoint()   
                self.current_iteration += 1
                # Save checkpoint after each epoch
            self.save_checkpoint()

    
    def save_checkpoint(self):
        checkpoint = {
            'sdf_model': self.sdf_model.state_dict(),
            'current_iteration': self.current_iteration,
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_path, 'checkpoint_{}_{}.pth'.format(self.curr_class_,self.current_iteration)))
        
    def load_checkpoint(self, checkpoint_name):
        print("*** Loading Checkpoint: ", checkpoint_name)
        checkpoint = torch.load(checkpoint_name, map_location=self.device)
        self.sdf_model.load_state_dict(checkpoint['sdf_model'])
        self.current_iteration = checkpoint['current_iteration']
    
    def get_features_add_sample(self, points):
        noisy_points = sample_query_points_entire_pc(points)
        noisy_points = torch.from_numpy(noisy_points).to(torch.float32)
        noisy_points = noisy_points.unsqueeze(0)
        points = points.to(self.device)
        noisy_points = noisy_points.to(self.device)
        point_feature, g_point = self.sdf_model(points, noisy_points)
        point_feature = point_feature.squeeze(0)
        return point_feature
    
    def get_features_predict(self, points):
        noisy_points = sample_query_points_entire_pc(points)
        noisy_points = torch.from_numpy(noisy_points).to(torch.float32)
        noisy_points = noisy_points.unsqueeze(0)
        points = points.to(self.device)
        noisy_points = noisy_points.to(self.device)
        point_feature = self.sdf_model.predict(points, noisy_points)
        point_feature = point_feature.squeeze(0)
        return point_feature


if __name__ == "__main__":
    # training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="./checkpoint_pointnet")
    parser.add_argument('--threed_backbone', type=str, default='pointnet')
    parser.add_argument('--category', type=str, default="diamond")
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--learning_rate', type=int, default=0.0001)
    parser.add_argument('--dataset_path', type=str, default='./datasets/Real_AD_3D_multiview/')

    # trained on single GPU
    cuda_idx = str(0)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx

    classes = os.listdir(parser.parse_args().dataset_path)
    # classes = ["airplane","candybar","car","chicken","diamond","duck"]

    parameter_dict = parser.parse_args()
    pointnet_version = parameter_dict.threed_backbone
    hyperparameters = {
        "epoch": parameter_dict.epoch,
        "learning_rate": parameter_dict.learning_rate,
        "classes": classes,
    }

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_name = f"{pointnet_version}_"
    checkpoint_path_global = os.path.join(parameter_dict.ckpt_path, checkpoint_name)

    for class_ in classes:
        # create the ckpt dir and parameter file
        checkpoint_path_local = checkpoint_path_global + f"{class_}_" + time
        if not osp.exists(checkpoint_path_local):
            os.makedirs(checkpoint_path_local)
        save(hyperparameters, os.path.join(checkpoint_path_local, "Congiguration"))
        hyperparameters["checkpoint_path"] = checkpoint_path_local

        print(" *** Training the 3D model *** ")
        print(" *** Hyperparameters *** ")
        print(hyperparameters)

        pretrain = Training(hyperparameters, class_, point_net_backbone=pointnet_version)
        
        img_path = os.path.join(parser.parse_args().dataset_path, class_, 'train', 'good')

        _224_pcd_paths = glob.glob(os.path.join(img_path, 'xyz', '*', '224_pcd.npy'))

        train_data_loader = get_train_data_loader(_224_pcd_paths)

        pretrain.train(train_data_loader)
        
    print(" *** Pre-training the model completed *** ")

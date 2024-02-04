import argparse
from patchcore_runner_cpmf import MultiViewPatchCore

def run_3d_ads(args):
    cls = args.category
    backbone_name = args.backbone

    
    if args.use_pointnet:
        point_net_backbone = 'pointnet'
        checkpoint_name = "/mnt/d/Project/cpmf_git/3D-Anomaly-Detection/CPMF/checkpoint_pointnet2/2024-02-04_13-57-28-pointnet/checkpoint_diamond_50.pth"
    elif args.use_pointnet2:
        point_net_backbone = 'pointnet2'
        checkpoint_name = None
    else:
        point_net_backbone = None
        checkpoint_name = None
        
    print('=========================================')
    kwargs = vars(args)
    for k, v in kwargs.items():
        print(f'{k}: {v}')
    print('=========================================')

    print(f"\n {args.exp_name} \n")
    print(f"\nRunning on class {cls}\n")
    patchcore = MultiViewPatchCore(backbone_name=backbone_name, dataset_path=args.data_path, n_views=args.n_views, no_fpfh=args.no_fpfh, point_net_backbone = point_net_backbone,
                                   class_name=cls, root_dir=args.root_dir, exp_name=args.exp_name, plot_use_rgb=args.use_rgb)

    ############## fit ###############
    patchcore.fit(checkpoint_name)

    ############# evaluate ###########
    patchcore.evaluate(checkpoint_name=checkpoint_name, draw=args.draw)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data-path', type=str, default='/mnt/d/Project/cpmf_git/3D-Anomaly-Detection/CPMF/datasets/Real_AD_3D_multi_view')
    parser.add_argument('--n-views', type=int, default=1)
    parser.add_argument('--no-fpfh', type=str2bool, default=False)
    parser.add_argument("--use-pointnet", type=str2bool, default=True)
    parser.add_argument("--use-pointnet2", type=str2bool, default=False)
    parser.add_argument('--use-rgb', type=str2bool, default=False)
    parser.add_argument('--exp-name', type=str, default='default')
    parser.add_argument('--category', type=str, default='diamond')
    parser.add_argument('--root-dir', type=str, default='./results')
    parser.add_argument('--backbone', type=str, default='nasnetalarge')
    parser.add_argument('--draw', type=str2bool, default=False)

    args = parser.parse_args()  
    
    run_3d_ads(args)

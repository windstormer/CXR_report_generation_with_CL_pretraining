import torch
import os
import numpy as np
from sklearn.decomposition import PCA
import cv2
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess import *
from dataset import *
from model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument("--dataset_path",
                        type=str, 
                        default="/hdd/vincent18/iu_xray/",
                        help="path of dataset")

    parser.add_argument("--gpu_id",
                        type=str,
                        default ='',
                        help="gpu id number")

    parser.add_argument("--patch_size",
                        "-p",
                        type=int,
                        default=224,
                        help="image size")

    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=32,
                        help="batch size")  
    
    parser.add_argument("--suffix",
                        "-s",
                        type=str,
                        default="",
                        help="suffix")

    args = parser.parse_args()
    if args.gpu_id != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    batch_size = args.batch_size
    patch_size = args.patch_size

    print("============== Load Dataset ===============")
    figure_path = '../figure/'
    model_type = 'SSL'
    if model_type == 'SSL':
        model = Model().cuda()
        model.load_state_dict(torch.load("../record/SSL_p224_ep250_b32/model/encoder_250.pth", map_location='cpu'), strict=False)
        train_dataset = TrainDataset(args, 'test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    elif model_type == 'AE':
        model = AutoEncoder().cuda()
        model.load_state_dict(torch.load("../record/AE_p224_ep250_b32/model/encoder_250.pth", map_location='cpu'), strict=False)
        train_dataset = TrainDataset(args, 'test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    elif model_type == 'imagenet':
        model = Model(pretrained='imagenet').cuda()
        train_dataset = TrainDataset(args, 'test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
        
    for param in model.parameters():
        param.requires_grad = False

    if args.suffix == "":
        model_type_suffix = model_type
    else:
        model_type_suffix = "{}_{}".format(model_type, args.suffix)
    
    
    data_loader = train_loader
    model.eval()
    data_bar = tqdm(data_loader)
    feature_list = []
    for img in data_bar:
        img = img.cuda(non_blocking=True)
        code, decode = model(img)
        feature_list.append(code.cpu())
        
    feature_list = torch.cat(feature_list)
    print("feature_list.shape", feature_list.shape)
    feature_list = feature_list.numpy()

    X = PCA(n_components=2).fit_transform(feature_list)
    print(X.shape)
    X_norm = (X - X.min()) / (X.max() - X.min())

    plt.figure(figsize=(10, 10))
    plt.scatter(X_norm[:,0], X_norm[:,1])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(os.path.join(figure_path, "PCA_{}.png".format(model_type_suffix)))

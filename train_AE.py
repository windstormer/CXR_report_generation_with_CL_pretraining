import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import cv2
import torch.nn as nn

from utils import *
from dataset import *
from model import AutoEncoder
from loss import *
# from preprocess import *

def train(args, epoch, net, data_loader, train_optimizer):
    net.train()
    train_bar = tqdm(data_loader)
    total_loss, total_num = 0.0, 0
    mseloss = nn.MSELoss()
    for img in train_bar:
        # print(img.max(), img.min())
        img = img.cuda(non_blocking=True)
        code, decode = net(img)
        # print(decode.max(), decode.min())
        loss = mseloss(decode, img)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num

def val(args, epoch, net, data_loader):
    net.eval()
    val_bar = tqdm(data_loader)
    total_loss, total_num = 0.0, 0
    mseloss = nn.MSELoss()
    with torch.no_grad(): 
        for img in val_bar:
            # print(img.max(), img.min())
            img = img.cuda(non_blocking=True)
            code, decode = net(img)
            # print(decode.max(), decode.min())
            loss = mseloss(decode, img)

            total_num += args.batch_size
            total_loss += loss.item() * args.batch_size
            val_bar.set_description('Val Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))
    return total_loss / total_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')

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

    parser.add_argument("--epochs",
                        "-e",
                        type=int,
                        default=250,
                        help="number of epoch")

    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=32,
                        help="batch size")          

    parser.add_argument("--suffix",
                        '-s',
                        type=str,
                        default=None,
                        help="suffix")
    
    # args parse
    args = parser.parse_args()
    if args.gpu_id != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    record_path = "../record"
    
    model_name = "AE_p{}_ep{}_b{}".format(args.patch_size, args.epochs, args.batch_size)    

    if args.suffix != None:
        model_name = model_name + ".{}".format(args.suffix)

    full_log_path = os.path.join(record_path, model_name, "{}.log".format(model_name))
    if not os.path.exists(os.path.join(record_path, model_name, "model")):
        os.makedirs(os.path.join(record_path, model_name, "model"))
    test_batch_size = 128

    log_file = open(full_log_path, "w+")
    log_file.writelines(str(datetime.now())+"\n")
    log_file.close()

    print("============== Load Dataset ===============")
    train_dataset = TrainDataset(args, 'train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_dataset = TrainDataset(args, 'val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    print("Total images:", len(train_dataset))
    # del image
    print("============== Model Setup ===============")
    # model setup and optimizer config
    model = AutoEncoder().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("============== Start Training ===============")
    # training loop
    record = {'train_loss':[], 'val_loss':[]}
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, epoch, model, train_loader, optimizer)
        record['train_loss'].append(train_loss)
        val_loss = val(args, epoch, model, val_loader)
        record['val_loss'].append(val_loss)
        log_file = open(full_log_path,"a")
        log_file.writelines("Epoch {:4d}/{:4d} | Train Loss: {:.5f} | Val Loss: {:.5f}\n".format(epoch, args.epochs, train_loss, val_loss))
        log_file.close()
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(record_path, model_name, "model", "encoder_{}.pth".format(epoch)))
    save_chart(args.epochs, record['train_loss'], record['val_loss'], os.path.join(record_path, model_name, "loss.png"), name='loss')

        
        


        


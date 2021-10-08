import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import cv2

from utils import *
from dataset import *
from model import Model
from loss import *
# from preprocess import *

def train(args, epoch, net, data_loader, train_optimizer):
    net.train()
    train_bar = tqdm(data_loader)
    total_loss, total_num = 0.0, 0
    for pos_1, pos_2 in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        loss = NTXent(out_1, out_2, args.batch_size, device=out_1.device)
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
    with torch.no_grad():
        for pos_1, pos_2 in val_bar:
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)

            loss = NTXent(out_1, out_2, pos_1.shape[0], device=out_1.device)

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
                        default=256,
                        help="image size")

    parser.add_argument("--epochs",
                        "-e",
                        type=int,
                        default=500,
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
    
    model_name = "SSL_p{}_ep{}_b{}".format(args.patch_size, args.epochs, args.batch_size)    

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
    train_dataset = SSLTrainDataset(args, 'train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_dataset = SSLTrainDataset(args, 'val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    print("Total images:", len(train_dataset))
    # del image
    print("============== Model Setup ===============")
    # model setup and optimizer config
    model = Model(args.feature_dim).cuda()
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
    

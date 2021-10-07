import argparse
import os

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import cv2

from dataset import *
from model import Model
from loss import *
from preprocess import *



def train(net, data_loader, train_optimizer):
    net.train()
    train_bar = tqdm(data_loader)
    total_loss, total_num = 0.0, 0
    for pos_1, pos_2 in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        loss = NTXent(out_1, out_2, batch_size, device=out_1.device)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')

    parser.add_argument("--dataset_path",
                        type=str, 
                        default="/hdd/vincent18/kaggle_dataset_data/",
                        help="path of dataset")

    parser.add_argument("--report_path",
                        type=str, 
                        default="/hdd/vincent18/IUXR_report/",
                        help="path of report")
                        
    # parser.add_argument("--img_size",
    #                     type=int,
    #                     nargs=2,
    #                     default=[512, 512],
    #                     help="image size [x, y]")

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
    feature_dim, temperature = args.feature_dim, args.temperature
    batch_size, epochs = args.batch_size, args.epochs
    suffix = args.suffix
    # img_size = args.img_size

    model_path = "model"
    log_path = "log"
    
    model_name = "Kaggle_Self_p{}_ep{}_b{}".format(args.patch_size, epochs, batch_size)    

    if suffix != None:
        model_name = model_name + ".{}".format(suffix)

    full_log_path = os.path.join(log_path, "{}.log".format(model_name))
    if not os.path.exists(os.path.join(model_path, model_name)):
        os.makedirs(os.path.join(model_path, model_name))
    test_batch_size = 128

    log_file = open(full_log_path, "w+")
    log_file.writelines(str(datetime.now())+"\n")
    log_file.close()

    print("============== Load Dataset ===============")
    # report, image_name_list = load_report(args.report_path)
    # image = load_image(args.dataset_path, image_name_list, args.img_size, patch_size)
    image = load_kaggle_image(args.dataset_path, args.patch_size)
    n_case = len(image)
    print("n_case", n_case)
    print("image.shape", image.shape)
    # train_data = load_patch_dataset(image, patch_size)

    train_dataset = SSLTrainDataset(image, args.patch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    # del image
    print("============== Model Setup ===============")
    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("============== Start Training ===============")
    # training loop
    best_f1 = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        log_file = open(full_log_path,"a")
        log_file.writelines("Epoch {:4d}/{:4d} | Train Loss: {}\n".format(epoch, epochs, train_loss))
        log_file.close()
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(model_path, model_name, "self_model_{}.pth".format(epoch)))
    

        
        


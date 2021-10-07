import argparse
import os

import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn

from dataset import *
from model import Model
from loss import *
from preprocess import *
from skimage import io
from skimage import img_as_ubyte

def train(net, data_loader, train_optimizer):
    # lossClass = SupConLoss()
    # lossClass = nn.MSELoss()
    net.train()
    train_bar = tqdm(data_loader)
    total_loss, total_num = 0.0, 0
    for pos_1, pos_2 in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        img1 = ((pos_1+1)/2).permute(0, 2, 3, 1).detach().cpu().numpy()
        img2 = ((pos_2+1)/2).permute(0, 2, 3, 1).detach().cpu().numpy()
        # seg1 = seg_1.permute(0, 2, 3, 1).detach().cpu().numpy()
        # seg2 = seg_2.permute(0, 2, 3, 1).detach().cpu().numpy()
        # if not os.path.exists(os.path.join("test_image", model_name)):
        #     os.makedirs(os.path.join("test_image", model_name))
        # for i in range(3):
        #     io.imsave(os.path.join("test_image", model_name, "img1_{}.png".format(i)), img_as_ubyte(img1[i, :, :, 0]), check_contrast=False)
        #     io.imsave(os.path.join("test_image", model_name, "img2_{}.png".format(i)), img_as_ubyte(img2[i, :, :, 0]), check_contrast=False)
        #     io.imsave(os.path.join("test_image", model_name, "seg1_{}.png".format(i)), img_as_ubyte(img1[i, :, :, 2]), check_contrast=False)
        #     io.imsave(os.path.join("test_image", model_name, "seg2_{}.png".format(i)), img_as_ubyte(img2[i, :, :, 2]), check_contrast=False)


        # loss = lossClass(out_1, out_2)
        # seg_1 = seg_1.cuda()
        # seg_2 = seg_2.cuda()
        # if attloss:
        #     ntxent = NTXent(out_1, out_2, batch_size, device=out_1.device)
        #     mseloss = lossClass(att1, seg_1.cuda()) + lossClass(att2, seg_2)
        #     loss = ntxent + 10*mseloss
        #     print("NTxent:", ntxent.detach().cpu().numpy(), "MSE:",  10*mseloss.detach().cpu().numpy())
        # else:
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
                        default="../IUXR_png/",
                        help="path of dataset")

    parser.add_argument("--report_path",
                        type=str, 
                        default="../IUXR_report/",
                        help="path of report")
                        
    parser.add_argument("--img_size",
                        type=int,
                        nargs=2,
                        default=[512, 512],
                        help="image size [x, y]")

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
    gpu_id = args.gpu_id
    if gpu_id == '':
        gpu_id = ",".join([str(g) for g in np.arange(torch.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpuid = range(len(gpu_id.split(",")))
    feature_dim, temperature = args.feature_dim, args.temperature
    batch_size, epochs = args.batch_size, args.epochs
    patch_size = args.patch_size
    suffix = args.suffix
    img_size = args.img_size

    model_path = "model"
    log_path = "log"
    
    model_name = "AttSSL_p{}_ep{}_b{}".format(patch_size, epochs, batch_size)    

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
    report, image_name_list = load_report(args.report_path)
    image, seg = load_image_seg(args.dataset_path, image_name_list, args.img_size, patch_size)
    n_case = len(report)
    print("n_case", n_case)
    print("image.shape", image.shape)
    print("seg.shape", seg.shape)
    # train_data = load_patch_dataset(image, patch_size)

    train_dataset = AttSSLTrainDataset(image, patch_size, seg)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_img = image[:10]
    val_seg = seg[:10]
    # del image
    print("============== Model Setup ===============")
    # model setup and optimizer config
    model = Model(feature_dim)
    if len(gpuid) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpuid)
    model = model.cuda()
    # flops, params = profile(model, inputs=(torch.randn(1, 3, patch_size, patch_size).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
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

        
        


import torch
import os
import numpy as np
import cv2
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from dataset import *
from model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tokenizers import *
from evaluation import compute_scores
from utils import *
from transformer_model import *


def train(cnet, dnet, train_loader, optimizer, strategy):
    if strategy == 'freeze':
        cnet.eval()
        dnet.train()
    else:
        cnet.train()
        dnet.train()
    train_bar = tqdm(train_loader)
    total_loss, total_num = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    # flag = False

    for case_id, img, caption, _, length in train_bar:
        img, caption = img.cuda(), caption.cuda()
        targets = caption.flatten()
        caption = torch.transpose(caption, 0, 1)
       

        target_caption = caption[:-1, :]
        tcaption_loss = caption[1:, :]
        mask = generate_square_subsequent_mask(target_caption.shape[0])
        
        
        features = cnet(img)   
        outputs = dnet(src=features, trg=target_caption, tgt_mask=mask)

        # if flag == False:
        #     tmp = torch.transpose(outputs, 0, 1)
        #     print(torch.argmax(tmp[0], dim=1))
        #     print(torch.transpose(target_caption, 0, 1)[0])
        #     flag = True
        
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tcaption_loss.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_num += len(img)
        total_loss += loss.item() * len(img)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))

    return total_loss / total_num

def val(cnet, dnet, val_loader, model_name, epoch):
    cnet.eval()
    dnet.eval()
    val_bar = tqdm(val_loader)

    criterion = nn.CrossEntropyLoss()
    total_loss, total_num = 0.0, 0
    with torch.no_grad():
        for case_id, img, caption, _, length in val_bar:
            img, caption = img.cuda(), caption.cuda()
            
            caption = torch.transpose(caption, 0, 1)
        

            target_caption = caption[:-1, :]
            tcaption_loss = caption[1:, :]
            mask = generate_square_subsequent_mask(target_caption.shape[0])
            
            
            features = cnet(img)   
            outputs = dnet(src=features, trg=target_caption, tgt_mask=mask)
            
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tcaption_loss.reshape(-1))

            total_num += len(img)
            total_loss += loss.item() * len(img)
            val_bar.set_description('Val Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))
        
    return total_loss / total_num


def test(cnet, dnet, test_loader, model_name, epoch):
    cnet.eval()
    dnet.eval()
    test_bar = tqdm(test_loader)

    criterion = nn.CrossEntropyLoss()
    total_loss, total_num = 0.0, 0
    with torch.no_grad():
        for case_id, img, caption, _, length in test_bar:
            img, caption = img.cuda(), caption.cuda()
            
            caption = torch.transpose(caption, 0, 1)
        

            target_caption = caption[:-1, :]
            tcaption_loss = caption[1:, :]
            mask = generate_square_subsequent_mask(target_caption.shape[0])
            
            
            features = cnet(img)   
            outputs = dnet(src=features, trg=target_caption, tgt_mask=mask)

            
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tcaption_loss.reshape(-1))

            total_num += len(img)
            total_loss += loss.item() * len(img)
            test_bar.set_description('Test Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, args.epochs, total_loss / total_num))
        
    return total_loss / total_num

def test_write_output(args, cnet, dnet, result_loader, result_path, model_name, epoch, tokenizer):
    cnet.eval()
    dnet.eval()
    test_bar = tqdm(result_loader, desc='Report Generating')
    max_caption_len = args.max_seq_length
    i=0
    result_file = open(os.path.join(result_path, model_name, "caption_{}.log".format(epoch)), "w+")
    result_file.writelines(str(datetime.now())+"\n")
    res, gts = [], []
    with torch.no_grad():
        for case_id, img, caption, _, __ in test_bar:
            img, caption = img.cuda(), caption.cuda()
            features = cnet(img)
            outputs = dnet.sample(memory=features, max_len=max_caption_len)
            reports = tokenizer.decode(outputs[0, 1:].cpu().numpy())
            ground_truth = tokenizer.decode(caption[0, 1:].cpu().numpy())
            res.append(reports)
            gts.append(ground_truth)

            if i < 30:
                result_file.writelines("======= {} =======\n".format(case_id))
                result_file.writelines("Out: {}\n".format(reports))
                result_file.writelines("True: {}\n".format(ground_truth))
            i+=1

    val_metric = compute_scores({i: [gt] for i, gt in enumerate(gts)}, {i: [re] for i, re in enumerate(res)})
    for k, v in val_metric.items():
        result_file.writelines("{}: {}\n".format(k, v))
    result_file.close()

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
                        default=256,
                        help="image size")

    parser.add_argument("--epochs",
                        "-e",
                        type=int,
                        default=200,
                        help="number of epoch")

    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=64,
                        help="batch size")          

    parser.add_argument("--suffix",
                        '-s',
                        type=str,
                        default=None,
                        help="suffix")
    
    parser.add_argument("--strategy",
                        type=str,
                        default='finetune',
                        choices=['finetune', 'freeze', 'scratch'],
                        help="training strategy")

    parser.add_argument("--pretrain",
                        type=str,
                        default=None,
                        help="pretrain model")

    parser.add_argument("--encoder_type",
                        type=str,
                        default="SSL",
                        help="encoder model type")

    parser.add_argument("--hyper",
                        type=int,
                        nargs=4,
                        default=[128, 128, 4, 4],
                        help="d_model, embed_size, nhead and layers")

    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')

    cell_type = "Transformer"

    args = parser.parse_args()
    if args.gpu_id != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    layers = args.hyper[3]
    nhead = args.hyper[2]
    embed_size = args.hyper[1]
    d_model = args.hyper[0]

    record_path = "../record"
    result_path = "../results"
    encoder_type = args.encoder_type
    if args.pretrain == "Imagenet":
        model_name = "Imagenet_{}_p{}_ep{}_b{}_{}".format(cell_type, args.patch_size, args.epochs, args.batch_size, args.strategy)
    elif args.pretrain == None:
        model_name = "{}_{}_p{}_ep{}_b{}_scratch".format(encoder_type, cell_type, args.patch_size, args.epochs, args.batch_size)
    else:
        encoder_type = args.pretrain.split("_")[0]
        model_name = "{}_{}_p{}_ep{}_b{}_{}".format(encoder_type, cell_type, args.patch_size, args.epochs, args.batch_size, args.strategy)
    
    if encoder_type == 'SegSSL' or encoder_type == 'Tag':
        encoder_type = 'SSL'

    if args.suffix != None:
        model_name = model_name + ".{}".format(args.suffix)

    full_log_path = os.path.join(record_path, model_name, "{}.log".format(model_name))
    if not os.path.exists(os.path.join(record_path, model_name, "model")):
        os.makedirs(os.path.join(record_path, model_name, "model"))
    if not os.path.exists(os.path.join(result_path, model_name)):
        os.makedirs(os.path.join(result_path, model_name))
    
    if args.strategy == "scratch":
        args.pretrain = None
    log_file = open(full_log_path, "w+")
    log_file.writelines(str(datetime.now())+"\n")
    log_file.close()
    print("============== Load Dataset ===============")
    
    tokenizer = Tokenizer(args)
    if (args.pretrain == None) or (args.pretrain == "Imagenet"):
        pretrain_path = args.pretrain
    else:
        pretrain_path = os.path.join(record_path, args.pretrain, "model", "encoder_500.pth")

    cnet = EncoderCNN(pretrain_path, encoder_type).cuda()

    train_dataset = RNNDataset(args, 'train', tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    val_dataset = RNNDataset(args, 'val', tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    test_dataset = RNNDataset(args, 'test', tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    num_vocab = len(tokenizer.token2idx)+1 # add index 0
    print("num_vocab", num_vocab)
    dnet = Seq2SeqTransformer(num_decoder_layers=layers,nhead=nhead,
                            emb_size=embed_size,tgt_vocab_size=num_vocab,
                            dim_feedforward=d_model).cuda()
    if args.strategy == "freeze":
        optimizer = optim.Adam(dnet.parameters(), lr=1e-3, weight_decay=1e-6)
        for param in cnet.parameters():
            param.requires_grad = False
    else:
        optimizer = optim.Adam(list(cnet.parameters())+list(dnet.parameters()), lr=1e-3, weight_decay=1e-6)


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5, last_epoch=-1)


    print("============== Start Training ===============")
    # training loop
    record = {'train_loss':[], 'val_loss':[]}
    for epoch in range(1, args.epochs + 1):
        train_loss = train(cnet, dnet, train_loader, optimizer, args.strategy)
        record['train_loss'].append(train_loss)
        val_loss = val(cnet, dnet, val_loader, model_name, epoch)
        record['val_loss'].append(val_loss)
        scheduler.step()
        log_file = open(full_log_path,"a")
        log_file.writelines("Epoch {:4d}/{:4d} | Train Loss: {}\n".format(epoch, args.epochs, train_loss))
        log_file.writelines("Epoch {:4d}/{:4d} | Val Loss: {}\n".format(epoch, args.epochs, val_loss))
        log_file.close()
        if epoch % 10 == 0:
            test_loss = test(cnet, dnet, test_loader, model_name, epoch)
            log_file = open(full_log_path,"a")
            log_file.writelines("Epoch {:4d}/{:4d} | Test Loss: {}\n".format(epoch, args.epochs, test_loss))
            test_write_output(args, cnet, dnet, test_loader, result_path, model_name, epoch, tokenizer)
            log_file.writelines("Save model at Epoch {:4d}/{:4d} | Test Loss: {}\n".format(epoch, args.epochs, test_loss))
            torch.save(cnet.state_dict(), os.path.join(record_path, model_name, "model", "encoder_{}.pth".format(epoch)))
            torch.save(dnet.state_dict(), os.path.join(record_path, model_name, "model", "decoder_{}.pth".format(epoch)))
            log_file.close()
    save_chart(args.epochs, record['train_loss'], record['val_loss'], os.path.join(record_path, model_name, "loss.png"), name='loss')
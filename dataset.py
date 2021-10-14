from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import numpy as np
import cv2
import random
import json
from tokenizers import *

Image.MAX_IMAGE_PIXELS = 933120000

class SegSSLTrainDataset(Dataset):
    def __init__(self, args, split):
        self.annotation = json.loads(open(os.path.join(args.dataset_path, 'annotation.json'), 'r').read())
        self.cases = self.annotation[split]
        self.split = split
        self.data = []
        self.seg = []
        for cases in self.cases:
            self.data.append(os.path.join(args.dataset_path, 'images', cases['image_path'][0]))
            self.data.append(os.path.join(args.dataset_path, 'images', cases['image_path'][1]))
            self.seg.append(os.path.join(args.dataset_path, 'seg', cases['image_path'][0]))
            self.seg.append(os.path.join(args.dataset_path, 'seg', cases['image_path'][1]))

        if split == 'train':
            self.transform = transforms.Compose([
            transforms.Resize((args.patch_size, args.patch_size)),
            transforms.RandomResizedCrop(args.patch_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((args.patch_size, args.patch_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = self.data[index]
        seg = self.seg[index]
        x = Image.open(img)
        seg_img = Image.open(seg)
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        pos_1 = self.transform(x)
        if self.split == 'train':
            seg_seed = np.random.randint(100)
            if seg_seed < 50:
                random.seed(seed)
                torch.manual_seed(seed)
                seg_1 = self.transform(seg_img)
                pos_1 = seg_1 * pos_1

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        pos_2 = self.transform(x)
        if self.split == 'train':
            seg_seed = np.random.randint(100)
            if seg_seed < 50:
                random.seed(seed)
                torch.manual_seed(seed)
                seg_2 = self.transform(seg_img)
                pos_2 = seg_2 * pos_2
        return (pos_1-0.5)/0.5, (pos_2-0.5)/0.5
    
    def __len__(self):
        return len(self.data)

class SSLTrainDataset(Dataset):
    def __init__(self, args, split):
        self.annotation = json.loads(open(os.path.join(args.dataset_path, 'annotation.json'), 'r').read())
        self.cases = self.annotation[split]
        self.data = []
        for cases in self.cases:
            self.data.append(os.path.join(args.dataset_path, 'images', cases['image_path'][0]))
            self.data.append(os.path.join(args.dataset_path, 'images', cases['image_path'][1]))

        if split == 'train':
            self.transform = transforms.Compose([
            transforms.Resize((args.patch_size, args.patch_size)),
            transforms.RandomResizedCrop(args.patch_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((args.patch_size, args.patch_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = self.data[index]
        x = Image.open(img)
        pos_1 = self.transform(x)
        pos_2 = self.transform(x)
        return (pos_1-0.5)/0.5, (pos_2-0.5)/0.5
    
    def __len__(self):
        return len(self.data)

class TrainDataset(Dataset):
    def __init__(self, args, split):
        self.annotation = json.loads(open(os.path.join(args.dataset_path, 'annotation.json'), 'r').read())
        self.cases = self.annotation[split]
        self.data = []
        for cases in self.cases:
            self.data.append(os.path.join(args.dataset_path, 'images', cases['image_path'][0]))
            self.data.append(os.path.join(args.dataset_path, 'images', cases['image_path'][1]))

        if split == 'train':
            self.transform = transforms.Compose([
            transforms.Resize((args.patch_size, args.patch_size)),
            transforms.RandomResizedCrop(args.patch_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((args.patch_size, args.patch_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = self.data[index]
        x = Image.open(img)
        img = self.transform(x)
        return (img-0.5)/0.5
    
    def __len__(self):
        return len(self.data)

class RNNDataset(Dataset):
    def __init__(self, args, split, tokenizer):
        self.annotation = json.loads(open(os.path.join(args.dataset_path, 'annotation.json'), 'r').read())
        self.cases = self.annotation[split]
        self.data = []
        self.caption = []
        self.id = []
        for cases in self.cases:
            self.id.append(cases['id'])
            self.id.append(cases['id'])
            self.data.append(os.path.join(args.dataset_path, 'images', cases['image_path'][0]))
            self.data.append(os.path.join(args.dataset_path, 'images', cases['image_path'][1]))
            self.caption.append(tokenizer(cases['report'])[:args.max_seq_length])
            self.caption.append(tokenizer(cases['report'])[:args.max_seq_length])

        self.transform = transforms.Compose([
        transforms.Resize((args.patch_size, args.patch_size)),
        transforms.ToTensor()])

    def __getitem__(self, index):
        
        img = self.data[index]
        x = Image.open(img)
        img = self.transform(x)
        case_id = self.id[index]
        caption = self.caption[index]
        return case_id, (img-0.5)/0.5, caption
    
    def __len__(self):
        return len(self.data)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    image_id, images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = np.zeros((len(captions), max(lengths)), dtype=int)

    for i, cap in enumerate(captions):
        targets[i, :len(cap)] = cap 
    return image_id, images, torch.LongTensor(targets), captions, lengths
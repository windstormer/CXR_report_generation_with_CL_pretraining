import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torch.nn.utils.rnn import pack_padded_sequence
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self, feature_dim=128, pretrained=False):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50(pretrained=pretrained).named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), 
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), 
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class AutoEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super(AutoEncoder, self).__init__()
        self.f = []
        for name, module in resnet50(pretrained=pretrained).named_children():
            if not (isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.Linear)):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.g = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(2048, 512, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Upsample(scale_factor=2, mode='nearest'),
                               nn.Conv2d(512, 512, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(512, 128, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Upsample(scale_factor=2, mode='nearest'),
                               nn.Conv2d(128, 128, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(128, 64, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Upsample(scale_factor=2, mode='nearest'),
                               nn.Conv2d(64, 64, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(64, 32, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Upsample(scale_factor=2, mode='nearest'),
                               nn.Conv2d(32, 32, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(32, 16, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Upsample(scale_factor=2, mode='nearest'),
                               nn.Conv2d(16, 16, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(16, 3, kernel_size=3, padding=1),
                               nn.Tanh())

    def forward(self, inputs):
        f_out = self.f(inputs)
        codes = self.avg(f_out).flatten(1)
        decoded = self.g(f_out)
        # print(decoded.shape)
        return codes, decoded

class EncoderCNN(nn.Module):
    def __init__(self, pretrained_path, model_type):
        super(EncoderCNN, self).__init__()
        self.model_type = model_type
        if pretrained_path == 'Imagenet':
            if model_type == 'SSL':
                self.f = Model(pretrained=True).f
            elif model_type == 'AE':
                self.f = AutoEncoder(pretrained=True).f
                self.f.add_module("AdaptiveAvgPool2d", nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        else:
            if model_type == 'SSL' or model_type == 'Moco':
                self.f = Model().f
            elif model_type == 'AE':
                self.f = AutoEncoder().f
                self.f.add_module("AdaptiveAvgPool2d", nn.AdaptiveAvgPool2d(output_size=(1, 1)))

            if pretrained_path != None:
                # self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
                print("Model restore from", pretrained_path)
                state_dict_weights = torch.load(pretrained_path)
                state_dict_init = self.state_dict()
                filtered_dict = OrderedDict()
                for (k, v) in state_dict_weights.items():
                    if "f." in k:
                        filtered_dict[k] = v
                new_state_dict = OrderedDict()
                for (k, v), (k_0, v_0) in zip(filtered_dict.items(), state_dict_init.items()):
                    name = k_0
                    new_state_dict[name] = v
                    print(k, k_0)
                self.load_state_dict(new_state_dict, strict=False)
    def forward(self, images):
        features = self.f(images)
        features = features.view(features.size(0), -1)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, cell_type, num_layers=1):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        if cell_type == 'GRU':
            self.cell = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif cell_type == 'LSTM':
            self.cell = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.drop_emb = nn.Dropout(p=0.1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions, lengths):
        captions = captions[:, :-1]
        embed = self.embedding_layer(captions)
        embed = self.drop_emb(embed)
        embed = torch.cat((features.unsqueeze(1), embed), dim=1)
        # packed = pack_padded_sequence(embed, lengths, batch_first=True) 
        cell_outputs, _ = self.cell(embed)
        # print(cell_outputs.shape)
        out = self.linear(self.dropout(cell_outputs.reshape(-1, cell_outputs.shape[2])))
        # out = self.linear(self.dropout(cell_outputs.data))
        
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_sentence = []
        inputs = inputs.unsqueeze(1)
        for i in range(max_len):
            cell_outputs, states = self.cell(inputs, states)
            cell_outputs = cell_outputs.squeeze(0)
            out = self.linear(cell_outputs)
            _, last_pick = out.max(1)
            output_sentence.append(last_pick)
            # print(last_pick.shape)
            inputs = self.embedding_layer(last_pick).unsqueeze(1)
        output_sentence = torch.stack(output_sentence, 1)
        return output_sentence

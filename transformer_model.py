import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1).cuda()
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_decoder_layers: int, nhead: int,
                 emb_size: int, tgt_vocab_size: int,
                 dim_feedforward: int, dropout:float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
       
        self.embed_size = emb_size
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead


        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_size, nhead=self.nhead,
                                                dim_feedforward=dim_feedforward, dropout=dropout)
        decoder_norm = nn.LayerNorm(self.embed_size, eps=1e-5)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.generator = nn.Linear(self.embed_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, self.embed_size)
        self.positional_encoding_tgt = PositionalEncoding(self.embed_size, dropout=dropout)
        self.positional_encoding_src = PositionalEncoding(self.embed_size, dropout=dropout)
    
    def forward(self, src: Tensor, trg: Tensor, tgt_mask: Tensor):
        memory = torch.reshape(src, (src.shape[0], src.shape[1]//self.embed_size, self.embed_size))
        memory = torch.transpose(memory, 0, 1)
        memory = self.positional_encoding_src(memory)
        tgt_emb = self.positional_encoding_tgt(self.tgt_tok_emb(trg))
        outs = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.generator(outs)


    def decode(self, tgt: Tensor, src: Tensor, tgt_mask: Tensor):
        memory = torch.reshape(src, (src.shape[0], src.shape[1]//self.embed_size, self.embed_size))
        memory = torch.transpose(memory, 0, 1)
        memory = self.positional_encoding_src(memory)
        tgt_emb = self.tgt_tok_emb(tgt)
        return self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)


    def sample(self, memory, max_len):
        memory = memory.cuda()
        ys = torch.ones(1, 1).fill_(0).type(torch.long).cuda() # start token
        for i in range(max_len-1):
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                        .type(torch.bool)).cuda()
            out = self.decode(tgt=ys, src=memory, tgt_mask=tgt_mask)
            prob = self.generator(out)
            _, next_word = torch.max(prob, dim = 2)

            next_word =  torch.squeeze(torch.transpose(next_word, 0, 1), dim=0)

            next_word = next_word[-1].item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type(torch.long).fill_(next_word).cuda()], dim=0)
        ys = torch.transpose(ys, 0, 1)
        return ys


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size).cuda()
        pos = torch.arange(0, maxlen).reshape(maxlen, 1).cuda()
        pos_embedding = torch.zeros((maxlen, emb_size)).cuda()
        pos_embedding[:, 0::2] = torch.sin(pos * den).cuda()
        pos_embedding[:, 1::2] = torch.cos(pos * den).cuda()
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
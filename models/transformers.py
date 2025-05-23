import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np

DEVICE = torch.device('cuda:4')

'''
CITATION 
Code based off of work from:
@article{kodialam2020deep,
      title={Deep Contextual Clinical Prediction with Reverse Distillation}, 
      author={Rohan S. Kodialam and Rebecca Boiarsky and Justin Lim and Neil Dixit and Aditya Sai and David Sontag},
      year={2020},
      eprint={2007.05611},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
'''

# ----------
# GeLU
# ----------

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# -------------------
# Layers & Attention
# -------------------

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(query.size(-1))

        if mask is not None:
            mask_shaped = torch.einsum(
                'bi,bj->bij', (mask, mask)
            ).unsqueeze(1).expand(scores.shape)
            scores = scores.masked_fill(mask_shaped == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden_size = 300, num_attention_heads = 4, dropout=0.3):
        super().__init__()
        assert hidden_size % num_attention_heads == 0

        self.d_k = hidden_size // num_attention_heads
        self.h = num_attention_heads


        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)
        
        self.attn = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)

        self.attn = attn
        
        return self.output_linear(x)

class SublayerConnection(nn.Module):

    def __init__(self, hidden_size=300,dropout=0.3):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):

    def __init__(self,hidden_size=300,dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.w_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x))))

# ------------------------------------------
# Transformer Block using the layers above
# ------------------------------------------

class TransformerBlock(nn.Module):

    def __init__(self, hidden, heads, dropout=0.3, tr_dropout=0.3):

        super().__init__()
        self.attention = MultiHeadedAttention(hidden, heads, tr_dropout)
        self.feed_forward = PositionwiseFeedForward(hidden, tr_dropout)
        self.input_sublayer = SublayerConnection(hidden, tr_dropout)
        self.output_sublayer = SublayerConnection(hidden, tr_dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        x = self.input_sublayer(
            x, lambda y: self.attention.forward(y, y, y, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
    

# -------------------
# VisitTransformer 
# -------------------

class VisitTransformer(torch.nn.Module):
    def __init__(self, max_days, 
                 num_concepts):
        super(VisitTransformer, self).__init__()
        
        #Parameters
        self.n_heads = 2
        self.dropout = 0.3
        self.attn_depth = 2
        
        #Embedding parameters
        self.embedding_dim = 300*self.n_heads
        self.max_days = max_days 
        self.num_concepts = num_concepts
        
        #Time_Embedder & Concept_Embedder
        self.time_embedder = torch.nn.Embedding(self.max_days+1, self.embedding_dim)
        self.concept_embedder = torch.nn.Embedding(self.num_concepts+1, self.embedding_dim)

        #Transformer
        self.tfs = torch.nn.ModuleList([
                TransformerBlock(self.embedding_dim, self.n_heads, self.dropout)
                for _ in range(self.attn_depth)
        ])        
        
    def forward(self, x, t):
        # Number of patients (batch size) and max_visits
        batch_size = len(x)
        max_visits = t.shape[1]
        
        # Creating the masks leveraging t's padding (-1)
        mask2d = t.ge(0)
        mask3d = mask2d.unsqueeze(2).repeat(1, 1, self.embedding_dim)
        
        # Embedding the concepts from x (summed for invariance to permutations of the codes)
        conc_emb = [torch.stack([sum(self.concept_embedder(i)) for i in x[j]]) for j in range(len(x))]
        concepts = torch.zeros(batch_size, max_visits, self.embedding_dim)
        for i in range(batch_size):
            k = conc_emb[i].shape[0]
            concepts[i,:k,:] = conc_emb[i]
            
        # Embedding the time markers from t (sinusoïdal approach)   
        time = self.time_embedder((t*mask2d).long()) * mask3d
       
        # Passing it all in the Transformer (the mask will be used in the Attention layer + should be 2D)
        # Both input and output z are of shape (batch_size x max_visits x embed dim)
        z = concepts.to(device=DEVICE) + time.to(device=DEVICE)
        for tf in self.tfs:
            z = tf(z, mask2d)
        
        # Final Mask
        z = z*mask3d
        return z
    

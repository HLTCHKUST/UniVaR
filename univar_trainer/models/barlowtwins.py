
from typing import Iterable, Dict
from torch import nn, Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    
    def __init__(self, model: SentenceTransformer, lambd: float = 0.0051):
        """
        :param model: SentenceTransformer model
        """
        super(BarlowTwins, self).__init__()
        self.model = model
        self.lambd = lambd

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, return_features=False):
        sentence_features1, sentence_features2 = tuple(sentence_features)
        embeddings_a = self.model(sentence_features1)['sentence_embedding']  # (bsz, hdim)
        embeddings_b = self.model(sentence_features2)['sentence_embedding']

        batch_size = embeddings_a.size(0)
        z1 = F.normalize(embeddings_a, dim=1)
        z2 = F.normalize(embeddings_b, dim=1)
        z1 = F.normalize(z1, dim=0)
        z2 = F.normalize(z2, dim=0)

        # empirical cross-correlation matrix
        #c = self.bn(z1).T @ self.bn(z2)
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        return loss if not return_features else (loss, embeddings_a, embeddings_b)

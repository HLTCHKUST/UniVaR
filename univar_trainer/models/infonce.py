from typing import Iterable, Dict
import numpy as np
import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Modified from sentence_transformers.losses.ContrastiveTensionLossInBatchNegatives
# Use single model instead of 2 encoder models initialized using the same weights
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveTensionLoss.py#L90C7-L162
class InfoNCE(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=cos_sim):
        """
        :param model: SentenceTransformer model
        """
        super(InfoNCE, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        #self.scale = scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(scale))

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, return_features=False):
        sentence_features1, sentence_features2 = tuple(sentence_features)
        embeddings_a = self.model(sentence_features1)['sentence_embedding']  # (bsz, hdim)
        embeddings_b = self.model(sentence_features2)['sentence_embedding']

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.logit_scale.exp()  # self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        loss =  (self.cross_entropy_loss(scores, labels) + self.cross_entropy_loss(scores.t(), labels))/2

        return loss if not return_features else (loss, embeddings_a, embeddings_b)

class ValueAwareInfoNCE(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, mse_scale: float = 200.0, similarity_fct=cos_sim):
        """
        :param model: SentenceTransformer model
        """
        super(ValueAwareSiameseContrastiveTensionLossInBatchNegatives, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.scale = scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(scale))
        self.mse_scale = mse_scale

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], no_value_labels: Tensor):
        sentence_features1, sentence_features2 = tuple(sentence_features)
        embeddings_a = self.model(sentence_features1)['sentence_embedding']  # (bsz, hdim)
        embeddings_b = self.model(sentence_features2)['sentence_embedding']

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.logit_scale.exp()  # self.scale
        ce_labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        siamese_loss = (self.cross_entropy_loss(scores, ce_labels) + self.cross_entropy_loss(scores.t(), ce_labels))/2
        
        no_value_loss, value_loss = 0, 0
        nv_mask = no_value_labels == 1
        if nv_mask.any():
            nv_embeddings_a, nv_embeddings_b = embeddings_a[nv_mask, :], embeddings_b[nv_mask, :]
            nv_labels = torch.zeros_like(nv_embeddings_a, device=nv_embeddings_a.device, requires_grad=False)
            no_value_loss = (self.mse_loss(nv_embeddings_a, nv_labels) + self.mse_loss(nv_embeddings_b, nv_labels))/2
         
        v_mask = no_value_labels == 0
        if v_mask.any():
            v_embeddings_a, v_embeddings_b = embeddings_a[v_mask, :], embeddings_b[v_mask, :]
            v_labels = torch.zeros_like(v_embeddings_a, device=v_embeddings_a.device, requires_grad=False)
            value_loss = ((1-self.mse_loss(v_embeddings_a, v_labels)) + (1-self.mse_loss(v_embeddings_b, v_labels)))/2
            
        # print('siamese_loss', siamese_loss)
        # print('no_value_loss', no_value_loss * self.mse_scale)
        # print('value_loss', value_loss * self.mse_scale)
        return (siamese_loss + (self.mse_scale * no_value_loss)  + (self.mse_scale * value_loss))/3
    
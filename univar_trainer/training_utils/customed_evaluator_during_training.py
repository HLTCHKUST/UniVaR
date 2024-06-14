# from https://github.com/UKPLab/sentence-transformers/issues/336#issuecomment-1107891765

import logging
import os
import csv
import numpy as np
from typing import List, Union
import math
from tqdm.autonotebook import trange

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device

import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


class ValueEvaluator(SentenceEvaluator):

    def __init__(self, loaders, loss_model: nn.Module = None, name: str = '', logs_writer: SummaryWriter = None,
                 show_progress_bar: bool = False, write_csv: bool = True):

        """
        Evaluate a model based on the loss function.
        The returned score is loss value.
        The results are written in a CSV and Tensorboard logs.
        :param loaders:  Dict of Data loader object
        :param loss_model: loss module object
        :param name: Name for the output
        :param logs_writer: tensorboard writer object 
        :param show_progress_bar: If true, prints a progress bar
        :param write_csv: Write results to a CSV file
        """

        self.loaders = loaders
        self.write_csv = write_csv
        self.logs_writer = logs_writer
        self.name = name
        self.loss_model = loss_model

        # move model to gpu:  lidija-jovanovska
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        loss_model.to(self.device)

        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "loss_evaluation" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "loss"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        self.loss_model.eval()

        loss_value = 0
        for loader_name, loader in self.loaders.items():
            loader.collate_fn = model.smart_batching_collate
            num_batches = len(loader)
            data_iterator = iter(loader)

            all_sentence_features = []
            all_labels = []

            with torch.inference_mode():
                for _ in trange(num_batches, desc="Iteration", smoothing=0.05, disable=not self.show_progress_bar):
                    sentence_features, labels = next(data_iterator) 
                    #move data to GPU: lidija-jovanovska
                    sentence_features = list(map(lambda batch: batch_to_device(batch, self.device), sentence_features))
                    loss, embeddings_a, embeddings_b = self.loss_model(sentence_features, labels, return_features=True)
                    loss_value += loss.item()

                    all_sentence_features.append(embeddings_a.cpu())
                    all_labels.append(labels.cpu())

            all_sentence_features = torch.cat(all_sentence_features, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()

            cmap = 'Spectral'

            # umap = self.plot_umap(all_sentence_features, all_labels, cmap)
            # umap.savefig(os.path.join(output_path, f"umap_{loader_name}_epoch{epoch}_steps{steps}.png"))

            tsne = self.plot_tsne(all_sentence_features, all_labels, cmap)
            tsne.savefig(os.path.join(output_path, f"tsne_{loader_name}_epoch{epoch}_steps{steps}.png"))

            final_loss = loss_value / num_batches
            if output_path is not None and self.write_csv:

                csv_path = os.path.join(output_path, self.csv_file)
                output_file_exists = os.path.isfile(csv_path)

                with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not output_file_exists:
                        writer.writerow(self.csv_headers)

                    writer.writerow([epoch, steps, final_loss])

                self.logs_writer.add_scalar(f'eval/{loader_name}_loss', final_loss, steps)
                # self.logs_writer.add_figure(f'eval/{loader_name}_umap', umap, steps)
                self.logs_writer.add_figure(f'eval/{loader_name}_tsne', tsne, steps)

            self.loss_model.zero_grad()
            self.loss_model.train()

        return final_loss
    

    def plot_umap(self, all_sentence_features, all_labels, cmap='Spectral'):

        print(f'Running UMAP on {len(all_sentence_features)} samples...')

        umap_embeddings = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine').fit_transform(all_sentence_features)
        ax = plt.figure(figsize=(7, 7))
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=all_labels, cmap=cmap, s=10, alpha=0.5)
        plt.axis('off')
        plt.tight_layout()

        return ax

    def plot_tsne(self, all_sentence_features, all_labels, cmap='Spectral'):

        print(f'Running TSNE on {len(all_sentence_features)} samples...')

        tsne_embeddings = TSNE(n_components=2).fit_transform(all_sentence_features)
        ax = plt.figure(figsize=(7, 7))
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=all_labels, cmap=cmap, s=10, alpha=0.5)
        plt.axis('off')
        plt.tight_layout()

        return ax
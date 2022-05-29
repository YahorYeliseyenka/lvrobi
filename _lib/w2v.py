import os
import gensim
import h3
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb 
import optuna
import torch.optim as optim
import networkx as nx

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch import nn
from typing import List, Tuple
from torchmetrics.functional import f1
from pytorch_lightning.loggers.wandb import WandbLogger
from IPython.display import clear_output
from _lib.settings import DATA_OSM_CITIES_DIR, DATA_W2V_TESTS_DIR, DATA_TRIPS_AS_HEXES_DIR, DATA_W2V_VECTORS_DIR, DATA_TRIPS_AS_HEXES_GRAPH_DIR


class GraphTypes:
    NONE=0
    GRAPH=1
    DIGRAPH=2


class H3NeighborDataset(Dataset):
    def __init__(self, data: pd.DataFrame, graph: nx.Graph, ds_split=1):
        self.data = data
        self.graph = None
        self.ds_split = ds_split

        if graph:
            self.data = data[data.index.isin(set(data.index).intersection(graph.nodes))]

            self.graph = graph.copy()
            for n in set(graph.nodes).difference(data.index):
                self.graph.remove_node(n)

        self.data_torch = torch.Tensor(self.data.to_numpy())

        self.inputs, self.contexts, self.input_h3, self.context_h3, self.positive_indexes = self.generate_new_dataset(self.data)
    
    def __len__(self):
        return int(len(self.inputs)*self.ds_split)

    def __getitem__(self, index):
        input = self.data_torch[self.inputs[index]]
        context = self.data_torch[self.contexts[index]]
        input_h3 = self.input_h3[index]
        neg_index = self.get_random_negative_index(input_h3)
        negative = self.data_torch[neg_index]
        y_pos = 1.0
        y_neg = 0.0

        context_h3 = self.context_h3[index]
        negative_h3 = self.data.index[neg_index]
        return input, context, negative, y_pos, y_neg, input_h3, context_h3, negative_h3

    def generate_new_dataset(self, data: pd.DataFrame):
        all_indices = set(data.index)

        inputs = []
        contexts = []
        input_h3 = []
        context_h3 = []
 
        positive_indexes = {}

        for i, (h3_index, hex_data) in tqdm(enumerate(data.iterrows()), total=len(data)):
            if self.graph:
                available_neighbors_h3 = set(self.graph.neighbors(h3_index))

                negative_excluded_h3 = available_neighbors_h3.copy()
                negative_excluded_h3.add(h3_index)
            else:
                negative_excluded_h3 = h3.k_ring(h3_index, 1)

                available_neighbors_h3 = negative_excluded_h3.copy()
                available_neighbors_h3.remove(h3_index)

            available_neighbors_h3 = list(available_neighbors_h3.intersection(all_indices))
            negative_excluded_h3 = list(negative_excluded_h3.intersection(all_indices))
            
            positive_indexes4hex = [data.index.get_loc(idx) for idx in negative_excluded_h3]
            contexts_indexes = [data.index.get_loc(idx) for idx in available_neighbors_h3]

            inputs.extend([i] * len(contexts_indexes))
            contexts.extend(contexts_indexes)
            positive_indexes[h3_index] = set(positive_indexes4hex)

            input_h3.extend([h3_index] * len(available_neighbors_h3))
            context_h3.extend(available_neighbors_h3)

        return np.array(inputs), np.array(contexts), np.array(input_h3), np.array(context_h3), positive_indexes

    def get_random_negative_index(self, input_h3):
        excluded_indexes = self.positive_indexes[input_h3]
        while True:
            negative = np.random.randint(0, len(self.data))
            if negative not in excluded_indexes:
                break
        return negative


# class H3NeighborDataset4DG(Dataset):
#     def __init__(self, data: pd.DataFrame, graph: nx.DiGraph, neg_samples=True):
#         self.neg_samples = neg_samples

#         self.data = data[data.index.isin(set(data.index).intersection(graph.nodes))]

#         self.graph = graph.copy()
#         for n in set(graph.nodes).difference(data.index):
#             self.graph.remove_node(n)

#         self.data_torch = torch.Tensor(self.data.to_numpy())
#         self.inputs, self.contexts, self.probabilities, self.inputs_h3, self.contexts_h3 = self.generate_dataset(self.data)
    
#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, index):
#         input = self.data_torch[self.inputs[index]]
#         context = self.data_torch[self.contexts[index]]
#         probability = round(self.probabilities[index], 2)
#         input_h3 = self.inputs_h3[index]
#         context_h3 = self.contexts_h3[index]
        
#         return input, context, probability, input_h3, context_h3

#     def generate_dataset(self, data: pd.DataFrame):
#         inputs = []
#         contexts = []
#         probabilities = []
#         inputs_h3 = []
#         context_h3 = []

#         for i, (h3_index, hex_data) in tqdm(enumerate(data.iterrows()), total=len(data)):
#             available_neighbors_h3 = list(self.graph.neighbors(h3_index))

#             negative_excluded_h3 = set(available_neighbors_h3.copy())
#             negative_excluded_h3.add(h3_index)

#             probabilities4hex = [self.graph.get_edge_data(h3_index, n)['weight'] for n in available_neighbors_h3]

#             contexts_indexes = [data.index.get_loc(idx) for idx in available_neighbors_h3]

#             # df = self.data[~self.data.index.isin(negative_excluded_h3)].sample(n=len(available_neighbors_h3))

#             # probabilities4hex.extend([0.0] * len(contexts_indexes))
            
#             # contexts_indexes.extend([data.index.get_loc(idx) for idx in set(df.index)])
#             # available_neighbors_h3.extend(set(df.index))

#             inputs.extend([i] * len(contexts_indexes))
#             contexts.extend(contexts_indexes)
#             probabilities.extend(probabilities4hex)
#             inputs_h3.extend([h3_index] * len(available_neighbors_h3))
#             context_h3.extend(available_neighbors_h3)

#         return np.array(inputs), np.array(contexts), np.array(probabilities), np.array(inputs_h3), np.array(context_h3)


class BinaryNN(pl.LightningModule):
    def __init__(self, encoder_sizes, optimizer, lrate):
        super().__init__()

        def create_layers(sizes: List[Tuple[int]]) -> nn.Sequential:
            layers = []
            for i, (input_size, output_size) in enumerate(sizes):
                linear = nn.Linear(input_size, output_size)
                nn.init.xavier_uniform_(linear.weight)
                layers.append(nn.Linear(input_size, output_size))
                if i != len(sizes)-1:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)
        
        sizes = list(zip(encoder_sizes[:-1], encoder_sizes[1:]))
        self.encoder = create_layers(sizes)
        self.optim = optimizer
        self.lrate = lrate

    def forward(self, Xt: torch.Tensor, Xc: torch.Tensor):
        Xt_em = self.encoder(Xt)
        Xc_em = self.encoder(Xc)
        scores = torch.mul(Xt_em, Xc_em).sum(dim=1)
        return scores

    def predict(self, Xt: torch.Tensor, Xc: torch.Tensor):
        probas = F.sigmoid(self(Xt, Xc))
        return probas

    def training_step(self, batch, batch_idx):
        Xt, Xc, Xn, y_pos, y_neg, *_ = batch
        scores_pos = self(Xt, Xc)
        scores_neg = self(Xt, Xn)

        scores = torch.cat([scores_pos, scores_neg])
        y = torch.cat([y_pos, y_neg])

        loss = F.binary_cross_entropy_with_logits(scores, y)
        f_score = f1(F.sigmoid(scores), y.int())
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_f1', f_score, on_step=True, on_epoch=True)

        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        Xt, Xc, Xn, y_pos, y_neg, *_ = batch
        scores_pos = self(Xt, Xc)
        scores_neg = self(Xt, Xn)

        scores = torch.cat([scores_pos, scores_neg])
        y = torch.cat([y_pos, y_neg])

        loss = F.binary_cross_entropy_with_logits(scores, y)
        f_score = f1(F.sigmoid(scores), y.int())
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_f1', f_score, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self.optim(self.parameters(), lr=self.lrate)


def read_osm_data(resolution, graph_type):
    df = pd.read_csv(f'{DATA_OSM_CITIES_DIR}/{resolution}.csv', sep=';')
    df.set_index('h3', inplace = True)
    df = df.drop(columns=['city'])
    graph = None

    if graph_type != GraphTypes.NONE:
        df_trips = pd.read_csv(f'{DATA_TRIPS_AS_HEXES_DIR}/{resolution}.csv', sep=';')
        df = df[df.index.isin(list(df_trips['hexid'].unique()))]
        if graph_type == GraphTypes.DIGRAPH:
            graph = nx.read_edgelist(f'{DATA_TRIPS_AS_HEXES_GRAPH_DIR}/{resolution}.csv', nodetype=str, data=(('weight',float),), create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(f'{DATA_TRIPS_AS_HEXES_GRAPH_DIR}/{resolution}.csv', nodetype=str, data=(('weight',float),))

    return df, graph


def train_w2v_model(project_name, dataset, max_epochs, batch_size, optimizer_name, learning_rate, first_layer_size, second_layer_size, final_vector_size=50, p4wdb={}):
    wandb_config = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': optimizer_name,
            'epochs': max_epochs,
            'first_layer_size': first_layer_size,
            'second_layer_size': second_layer_size,
            'final_vector_size': final_vector_size
        }
    wandb_config.update(p4wdb)

    logger = WandbLogger()

    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, accelerator='gpu', logger=logger)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    leyers_size = [first_layer_size, second_layer_size, final_vector_size]
    model = BinaryNN(
                [dataset.data.shape[1], *leyers_size], 
                getattr(optim, optimizer_name), 
                learning_rate
            )

    model.cuda()

    run = wandb.init(
            # reinit=True, 
            project=project_name, 
            config=wandb_config,
            entity='bjuggler'
        )

    trainer.fit(model, train_dataloader)
    run.finish()

    return model, trainer


def run_w2v_tests(resolution, project_name, graph_type, ds_split=1, n_trials=100, final_vector_size=50, max_epochs=25):
    def objective(trial):
        trainer__optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'Adagrad', 'SGD']) 
        trainer__learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-0) 
        trainer__batch_size = trial.suggest_int('batch_size', 16, 512, step=16) 
        trainer__first_layer_size = trial.suggest_categorical('first_layer_size', [100, 125, 150, 175, 200])
        trainer__second_layer_size = trial.suggest_categorical('second_layer_size', [75, 100, 125, 150]) 

        model, trainer = train_w2v_model(project_name,
                                        dataset,
                                        max_epochs,
                                        trainer__batch_size,
                                        trainer__optimizer_name,
                                        trainer__learning_rate,
                                        trainer__first_layer_size,
                                        trainer__second_layer_size,
                                        final_vector_size)

        clear_output(wait=True)

        return float(trainer.callback_metrics['train_loss'])

    df, graph = read_osm_data(resolution, graph_type)
    dataset = H3NeighborDataset(df, graph, ds_split)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print('params:', study.best_params)
    print('value:', study.best_value)

    study_df = study.trials_dataframe()
    study_df.to_csv(f"{DATA_W2V_TESTS_DIR}/{project_name}.csv")


def save_vectors(file_name, model, df):
    embedded = model.encoder(torch.Tensor(df.to_numpy())).detach().numpy()

    embedded_df = pd.DataFrame(embedded)
    embedded_df = embedded_df.set_index(df.index)

    embedded_df.to_csv(f'{DATA_W2V_VECTORS_DIR}/{file_name}.csv', sep=' ', header=False)


def csv2kv(path2vectors_file, path2save):
    df = pd.read_csv(path2vectors_file, sep=' ', header=None)

    firstline = f'{df.shape[0]} {df.shape[1]-1}'
    filename = path2vectors_file.split('/')[-1].split('.')[0]
    word2vec_format_file_path = path2save.joinpath(filename+'.txt')

    with open(path2vectors_file, 'r') as fin, open(word2vec_format_file_path, 'w+') as fout:
        content = fin.read()
        fin.seek(0, 0)
        fout.write(firstline.rstrip('\r\n') + '\n' + content)
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(word2vec_format_file_path, binary=False)
    word_vectors.save(f"{path2save}/{filename}.kv")

    os.remove(word2vec_format_file_path)
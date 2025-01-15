# base
import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from functools import partial
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, average_precision_score
from dgl import batch
from dgllife.utils import smiles_to_bigraph
import concurrent.futures
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import sys
# from utils.cell_vit import cell_encoder
# from utils.cell_vit import TrainDataset
from QFormer.models import build_model
from QFormer.config import get_config_from_str

# for GraphEncoder (MolecularEncoder)
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy.stats import pearsonr, spearmanr
from dgllife.utils import atom_type_one_hot, atom_formal_charge, atom_hybridization_one_hot, atom_chiral_tag_one_hot, atom_is_in_ring, atom_is_aromatic, ConcatFeaturizer, BaseAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from dgllife.model.gnn.wln import WLN
from dgllife.model.gnn.gat import GAT
from dgllife.model.gnn.gcn import GCN
from dgllife.model.gnn.graphsage import GraphSAGE
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout.mlp_readout import MLPNodeReadout
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
import torch.nn.init as init

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")
    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")
    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')
    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys
        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)
        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)
        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.
        # Cosine between all combinations
        logits = query @ transpose(positive_key)
        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

class encoding2score(nn.Module):
    def __init__(self, 
            n_embed=1024, 
            dropout_rate=0.1, 
            expand_ratio=3,
            activation='GELU',
            initialize=False
        ):
        super(encoding2score, self).__init__()
        activation_dict = {
            'GELU': nn.GELU(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'SiLU': nn.SiLU(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU(),
            'CELU': nn.CELU(),
            'PReLU': nn.PReLU(),
        }
        if expand_ratio == 0:
            self.decoder = nn.Sequential(
                    nn.Linear(n_embed*2, 256, bias=True), # 1
                    nn.BatchNorm1d(256), # 2
                    activation_dict[activation],
                    nn.Dropout(p=dropout_rate, inplace=False),
                    nn.Linear(256, 1, bias=True),
                    nn.Identity()
                )
        else:
            self.decoder = nn.Sequential(
                    nn.Linear(n_embed*2, n_embed*expand_ratio*2), # 1
                    activation_dict[activation],
                    nn.BatchNorm1d(n_embed*expand_ratio*2), # 3
                    nn.Dropout(p=dropout_rate, inplace=False),
                    nn.Linear(n_embed*expand_ratio*2, 1024), # 5
                    activation_dict[activation],
                    nn.Linear(1024, 128), # 7
                    activation_dict[activation],
                    nn.Dropout(p=dropout_rate, inplace=False),
                    nn.Linear(128, 128), # 10
                    activation_dict[activation],
                    nn.Dropout(p=dropout_rate, inplace=False),
                    nn.Linear(128, 128), # 13
                    activation_dict[activation],
                    nn.Dropout(p=dropout_rate, inplace=False),
                    nn.Linear(128, 128), # 16
                    activation_dict[activation],
                    nn.Linear(128, 1),
                    nn.Identity(),
                )
        if initialize:
            self.decoder.apply(initialize_weights)
        self.cuda()
        
    def forward(self, features):
        score = self.decoder(features)
        prediction = torch.sigmoid(score).squeeze(-1)
        return prediction

class SkipConnection(nn.Module):
    def __init__(self, feature_size, skip_layers=2, activation=nn.GELU(), norm=None, dropout=0.0):
        super().__init__()
        self.skip_layers = skip_layers
        if norm == 'LayerNorm':
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.LayerNorm(feature_size),
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(skip_layers)
            ])
        elif norm == 'BatchNorm':
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.BatchNorm1d(feature_size),
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(skip_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(skip_layers)
            ])
        self.ident = nn.Identity()
        self.cuda()
    
    def forward(self, x):
        ident = self.ident(x)
        for i in range(self.skip_layers):
            x = self.layers[i](x)
        return torch.cat([ident, x], dim=-1)

class MLP(nn.Module):
    def __init__(self, feature_size, layers=2, activation=nn.GELU(), norm=None, dropout=0.0):
        super().__init__()
        self.num_layers = layers
        if norm == 'LayerNorm':
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.LayerNorm(feature_size),
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(layers)
            ])
        elif norm == 'BatchNorm':
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.BatchNorm1d(feature_size),
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(layers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(layers)
            ])
        self.cuda()

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x

class RugbyMLP(nn.Module):
    def __init__(self, feature_size, layers=2, activation=nn.GELU(), norm=None, dropout=0.0):
        super().__init__()
        self.num_layers = layers
        if norm == 'LayerNorm':
            self.layers = nn.Sequential(
                    nn.Linear(feature_size, feature_size*self.num_layers),
                    activation,
                    nn.LayerNorm(feature_size*self.num_layers),
                    nn.Dropout(p=dropout, inplace=False),
                    nn.Linear(feature_size*self.num_layers, feature_size*self.num_layers),
                    activation,
                    nn.LayerNorm(feature_size*self.num_layers),
                )
        elif norm == 'BatchNorm':
            self.layers = nn.Sequential(
                    nn.Linear(feature_size, feature_size*self.num_layers),
                    activation,
                    nn.BatchNorm1d(feature_size*self.num_layers),
                    nn.Dropout(p=dropout, inplace=False),
                    nn.Linear(feature_size*self.num_layers, feature_size*self.num_layers),
                    activation,
                    nn.BatchNorm1d(feature_size*self.num_layers)
                )
        else:
            self.layers = nn.Sequential(
                    nn.Linear(feature_size, feature_size*self.num_layers),
                    activation,
                    nn.Dropout(p=dropout, inplace=False),
                    nn.Linear(feature_size*self.num_layers, feature_size*self.num_layers),
                    activation,
                )
        self.cuda()

    def forward(self, x):
        return self.layers(x)

class GeminiMLP(nn.Module):
    def __init__(self, input_size, output_size, mlp_type='MLP', num_layers=3, activation='GELU', norm=None, dropout=0.0):
        super().__init__()
        activation_dict = {
            'GELU': nn.GELU(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'SiLU': nn.SiLU(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU(),
            'CELU': nn.CELU(),
            'Softplus': nn.Softplus(),
            'Softsign': nn.Softsign(),
            'Hardshrink': nn.Hardshrink(),
            'Hardtanh': nn.Hardtanh(),
            'LogSigmoid': nn.LogSigmoid(),
            'Softshrink': nn.Softshrink(),
            'PReLU': nn.PReLU(),
            'Softmin': nn.Softmin(),
            'Softmax': nn.Softmax()            
        }
        try:
            activation = activation_dict[activation]
        except:
            raise ValueError(f"Error: undefined activation function {activation}.")
        if mlp_type == 'MLP':
            self.gemini_mlp = MLP(
                input_size, 
                layers=num_layers, 
                activation=activation, 
                norm=norm, 
                dropout=dropout
            )
            self.out = nn.Linear(input_size, output_size)
        elif mlp_type == 'SkipConnection':
            self.gemini_mlp = SkipConnection(
                input_size, 
                skip_layers=num_layers, 
                activation=activation, 
                norm=norm, 
                dropout=dropout
            )
            self.out = nn.Linear(2*input_size, output_size)
        elif mlp_type == 'RugbyMLP':
            self.gemini_mlp = RugbyMLP(
                input_size, 
                layers=num_layers, 
                activation=activation, 
                norm=norm, 
                dropout=dropout
            )
            self.out = nn.Linear(input_size*num_layers, output_size)
        self.cuda()
    
    def forward(self, x):
        x = self.gemini_mlp(x)
        return self.out(x)

class SelfAttentionPooling(nn.Module):
    def __init__(self, attention_size=16):
        super().__init__()
        self.attention_size = attention_size
        self.cuda()
    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape
        attention_scores = torch.tanh(nn.Linear(embedding_size, self.attention_size, bias=False).cuda()(x))
        attention_weights = torch.softmax(nn.Linear(self.attention_size, 1, bias=False).cuda()(attention_scores), dim=1)
        pooled = (x * attention_weights).sum(dim=1)
        return pooled

def params2layerlist(params, layer_num=6):
    params_list = []
    for _ in range(layer_num):
        params_list.append(params) 
    return params_list

class MolecularEncoder(nn.Module):
    '''
    MolecularEncoder

    This is a graph encoder model consisting of a graph neural network from 
    DGL and the MLP architecture in pytorch, which builds an understanding 
    of a compound's conformational space by supervised learning of CSS data.

    '''
    def __init__(self,
        atom_feat_size = None, 
        bond_feat_size = None,
        num_features = 512,
        num_layers = 12,
        num_out_features = 1024,
        activation = 'GELU',
        readout_type = 'Weighted',
        integrate_layer_type = 'MLP',
        integrate_layer_num = 3,
        gnn_type = 'WLN'
        ):
        super().__init__()
        activation_dict = {
            'GELU': nn.GELU(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'SiLU': nn.SiLU(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU(),
            'CELU': nn.CELU(),
            'PReLU': nn.PReLU(),
        }
        # init the GeminiEncoder
        self.gnn_type = gnn_type
        if gnn_type == 'WLN':
            self.GeminiEncoder = WLN(
                atom_feat_size, 
                bond_feat_size, 
                n_layers=num_layers, 
                node_out_feats=num_features
            )
        elif gnn_type == 'GAT':
            self.GeminiEncoder = GAT(
                atom_feat_size, 
                hidden_feats=params2layerlist(num_features, layer_num=num_layers), 
                num_heads=params2layerlist(16, layer_num=num_layers), 
                feat_drops=params2layerlist(0, layer_num=num_layers), 
                activations=params2layerlist(nn.LeakyReLU(), layer_num=num_layers), 
                allow_zero_in_degree=True
            )
        elif gnn_type == 'GCN':
            self.GeminiEncoder = GCN(
                atom_feat_size, 
                hidden_feats=params2layerlist(num_features, layer_num=num_layers), 
                gnn_norm=params2layerlist('both', layer_num=num_layers), 
                dropout=params2layerlist(0, layer_num=num_layers), 
                allow_zero_in_degree=True
            )
        elif gnn_type == 'GraphSAGE':
            self.GeminiEncoder = GraphSAGE(
                atom_feat_size, 
                hidden_feats=params2layerlist(num_features, layer_num=num_layers), 
                dropout=params2layerlist(0, layer_num=num_layers)
            )
        elif gnn_type == 'GCNSAGE':
            self.GeminiEncoder = GraphSAGE(
                atom_feat_size, 
                hidden_feats=params2layerlist(num_features, layer_num=num_layers), 
                dropout=params2layerlist(0, layer_num=num_layers), 
                aggregator_type=params2layerlist('gcn', layer_num=num_layers)
            )
        elif gnn_type == 'AttentiveFP':
            self.GeminiEncoder = AttentiveFPGNN(
                atom_feat_size, 
                bond_feat_size, 
                num_layers=num_layers, 
                graph_feat_size=num_features, dropout=0.0
            )
        # init the readout and output layers
        self.readout_type = readout_type
        if readout_type == 'Weighted':
            assert num_features*2 == num_out_features, "Error: num_features*2 must equal num_out_features for Weighted readout."
            self.readout = WeightedSumAndMax(num_features)
        elif readout_type == 'MeanMLP':
            assert num_features == num_out_features, "Error: num_features must equal num_out_features for MLP readout."
            self.readout = MLPNodeReadout(
                num_features, num_features, num_features,
                activation=activation_dict[activation],
                mode='mean'
            )
        elif readout_type == 'ExtendMLP':
            self.readout = MLPNodeReadout(
                num_features, num_out_features, num_out_features,
                activation=activation_dict[activation],
                mode='mean'
            )
        elif readout_type == 'MMLP':
            self.readout = MLPNodeReadout(
                num_features, num_features, num_features,
                activation=activation_dict[activation],
                mode='mean'
            )
            self.output = GeminiMLP(
                num_features, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation,
                norm='BatchNorm'
            )
            self.output.cuda()
        elif readout_type == 'AttentiveFP':
            assert num_features == num_out_features, "Error: num_features must equal num_out_features for AttentiveFP readout."
            self.readout = AttentiveFPReadout(
                    num_features, 
                    dropout=0.0
                    )
        elif readout_type == 'AttentiveMLP':
            self.readout = AttentiveFPReadout(
                    num_features, 
                    dropout=0.0
                    )
            self.output = GeminiMLP(
                num_features, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation,
                norm='BatchNorm'
            )
            self.output.cuda()
        elif readout_type == 'WeightedMLP':
            self.readout = WeightedSumAndMax(num_features)
            self.output = GeminiMLP(
                num_features*2, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation,
                norm='BatchNorm'
            )
            self.output.cuda()
        elif readout_type == 'Mixed':
            self.readout = nn.ModuleDict({
                'Weighted': WeightedSumAndMax(num_features),
                'MLP': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='mean'
                )
            })
            self.output = GeminiMLP(
                num_features*3, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation 
            )
            self.output.cuda()
        elif readout_type == "MixedBN":
            self.readout = nn.ModuleDict({
                'Weighted': WeightedSumAndMax(num_features),
                'MLP': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='mean'
                )
            })
            self.output = GeminiMLP(
                num_features*3, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation,
                norm='BatchNorm'
            )
            self.output.cuda()
        elif readout_type == 'CombineMLP':
            assert num_features*3 == num_out_features, "Error: num_features must equal num_out_features for MLP readout."
            self.readout = nn.ModuleDict({
                'Max': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='max'
                ),
                'Sum': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='sum'
                ),
                'Mean': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='mean'
                )
            })
        elif readout_type == 'MixedMLP':
            self.readout = nn.ModuleDict({
                'Max': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='max'
                ),
                'Sum': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='sum'
                ),
                'Mean': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='mean'
                )
            })
            self.output = GeminiMLP(
                num_features*3, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation 
            )
            self.output.cuda()
        elif readout_type == 'CombineWeighted':
            assert num_features*3 == num_out_features, "Error: num_features must equal num_out_features for MLP readout."
            self.readout = nn.ModuleDict({
                'Weighted': WeightedSumAndMax(num_features),
                'MLP': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='mean'
                )
            })
        else:
            raise ValueError(f"Error: undefined readout type {readout_type}.")
        self.readout.cuda()
        self.GeminiEncoder.to('cuda')

    def forward(self, mol_graph):
        if self.gnn_type in ['WLN', 'AttentiveFP']:
            encoding = self.GeminiEncoder(mol_graph, mol_graph.ndata['atom_type'], mol_graph.edata['bond_type'])
        elif self.gnn_type in ['GAT', 'GCN', 'GraphSAGE', 'GCNSAGE']:
            encoding = self.GeminiEncoder(mol_graph, mol_graph.ndata['atom_type'])
        if self.readout_type in ['MixedBN', 'Mixed']:
            mixed_readout = (self.readout['Weighted'](mol_graph, encoding), self.readout['MLP'](mol_graph, encoding))
            return self.output(torch.cat(mixed_readout, dim=1))
        elif self.readout_type in ['AttentiveMLP', 'WeightedMLP', 'MMLP']:
            return self.output(self.readout(mol_graph, encoding))
        elif self.readout_type == 'MixedMLP':
            mixed_readout = (self.readout['Max'](mol_graph, encoding), self.readout['Sum'](mol_graph, encoding), self.readout['Mean'](mol_graph, encoding))
            return self.output(torch.cat(mixed_readout, dim=1))
        elif self.readout_type == 'CombineMLP':
            return torch.cat([self.readout['Max'](mol_graph, encoding), self.readout['Sum'](mol_graph, encoding), self.readout['Mean'](mol_graph, encoding)], dim=1)
        elif self.readout_type == 'CombineWeighted':
            return torch.cat([self.readout['Weighted'](mol_graph, encoding), self.readout['MLP'](mol_graph, encoding)], dim=1)
        else:
            return self.readout(mol_graph, encoding)

class BinarySimilarity(nn.Module):  
    def __init__(self, 
        geminimol_model_name, 
        feature_list = ['smiles1','smiles2'], 
        label_dict = {
            'ShapeScore':0.4, 
            'ShapeAggregation':0.2,
            'CrossSim':0.2, 
            'CrossAggregation':0.2
        }, 
        batch_size = 128, 
        gnn_type = 'WLN',
        num_layers = 12,
        num_features = 512,
        encoder_activation = 'GELU',
        readout_type = 'Mixed',
        geminimol_encoding_features = 1024,
        decoder_activation = 'GELU',
        decoder_expand_ratio = 3,
        decoder_dropout_rate = 0.1,
        integrate_layer_type = 'MLP',
        integrate_layer_num = 0,
        ):
        super(BinarySimilarity, self).__init__()
        torch.set_float32_matmul_precision('high') 
        self.geminimol_model_name = geminimol_model_name
        self.batch_size = batch_size
        self.label_list = list(label_dict.keys())
        self.feature_list = feature_list
        ## set loss weight
        loss_weight = list(label_dict.values())
        sum_weight = np.sum(loss_weight)
        self.loss_weight = {key: value/sum_weight for key, value in zip(self.label_list, loss_weight)}
        ## set up featurizer
        self.atom_featurizer = BaseAtomFeaturizer({'atom_type':ConcatFeaturizer([atom_type_one_hot, atom_hybridization_one_hot, atom_formal_charge, atom_chiral_tag_one_hot, atom_is_in_ring, atom_is_aromatic])}) 
        self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field='bond_type')
        ## create MolecularEncoder
        self.Encoder = MolecularEncoder(
            atom_feat_size=self.atom_featurizer.feat_size(feat_name='atom_type'), 
            bond_feat_size=self.bond_featurizer.feat_size(feat_name='bond_type'), 
            gnn_type=gnn_type,
            num_layers=num_layers,
            num_features=num_features,
            activation=encoder_activation,
            readout_type=readout_type,
            integrate_layer_type=integrate_layer_type,
            integrate_layer_num=integrate_layer_num,
            num_out_features = geminimol_encoding_features
        )
        # create multiple decoders
        self.Decoder = nn.ModuleDict()
        for label in self.label_list:
            self.Decoder[label] = encoding2score(
                n_embed=geminimol_encoding_features, 
                dropout_rate=decoder_dropout_rate,
                expand_ratio=decoder_expand_ratio,
                activation=decoder_activation
                )
        if os.path.exists(self.geminimol_model_name):
            if os.path.exists(f'{self.geminimol_model_name}/GeminiMol.pt'):
                self.load_state_dict(torch.load(f'{self.geminimol_model_name}/GeminiMol.pt'))
        else:
            os.mkdir(self.geminimol_model_name)
        

    def sents2tensor(self, input_sents):
        input_tensor = batch([smiles_to_bigraph(smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer).to('cuda') for smiles in input_sents])
        return input_tensor

    def forward(self, sent1, sent2, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # Concatenate input sentences
        input_sents = sent1 + sent2
        input_tensor = self.sents2tensor(input_sents).to('cuda')
        # Encode all sentences using the encoder
        features = self.Encoder(input_tensor)
        encoded1 = features[:batch_size]
        encoded2 = features[batch_size:]
        encoded = torch.cat((encoded1, encoded2), dim=1)
        # Calculate specific score between encoded smiles
        pred = {}
        for label in self.label_list:
            pred[label] = self.Decoder[label](encoded)
        return pred

    def encode(self, input_sents):
        # Encode all sentences using the encoder 
        input_tensor = self.sents2tensor(input_sents)
        features = self.Encoder(input_tensor)
        return features

    def extract_hidden_decoder_features(self, features, label_name, num_layers=3):
        if label_name in self.label_list:
            binrary_features = torch.cat((features.clone(), features.clone()), dim=1)
        else:
            binrary_features = features.clone()
        for i in range(num_layers):
            layer = self.Decoder[label_name].decoder[i] 
            binrary_features = layer(binrary_features)
        return binrary_features

    def decode(self, features, label_name, batch_size, depth=0):
        # CSS similarity
        if label_name in self.label_list:
            encoded = torch.cat((features[:batch_size], features[batch_size:]), dim=1)
            return self.Decoder[label_name](encoded)
        # re-shape encoded features
        if depth == 0:
            encoded1 = features[:batch_size]
            encoded2 = features[batch_size:]
        else:
            encoded1 = torch.cat([self.extract_hidden_decoder_features(features[:batch_size], label_name, num_layers=depth) for label_name in self.label_list], dim=1)
            encoded2 = torch.cat([self.extract_hidden_decoder_features(features[batch_size:], label_name, num_layers=depth) for label_name in self.label_list], dim=1)
        # decode vector similarity
        if label_name == 'RMSE':
            return 1 - torch.sqrt(torch.mean((encoded1.view(-1, encoded1.shape[-1]) - encoded2.view(-1, encoded2.shape[-1])) ** 2, dim=-1))
        elif label_name == 'Cosine':
            encoded1_normalized = torch.nn.functional.normalize(encoded1, dim=-1)
            encoded2_normalized = torch.nn.functional.normalize(encoded2, dim=-1)
            return torch.nn.functional.cosine_similarity(encoded1_normalized, encoded2_normalized, dim=-1)
        elif label_name == 'Euclidean':
            return 1 - torch.sqrt(torch.sum((encoded1 - encoded2)**2, dim=-1))
        elif label_name == 'Manhattan':
            return 1 - torch.sum(torch.abs(encoded1 - encoded2), dim=-1)
        elif label_name == 'Minkowski':
            return 1 - torch.norm(encoded1 - encoded2, p=3, dim=-1)
        elif label_name == 'KLDiv':
            encoded1_normalized = torch.nn.functional.softmax(encoded1, dim=-1)
            encoded2_log_normalized = torch.nn.functional.log_softmax(encoded2, dim=-1)
            return 1 - (torch.sum(encoded1_normalized * (torch.log(encoded1_normalized + 1e-7) - encoded2_log_normalized), dim=-1))
        elif label_name == 'Pearson':
            mean1 = torch.mean(encoded1, dim=-1, keepdim=True)
            mean2 = torch.mean(encoded2, dim=-1, keepdim=True)
            std1 = torch.std(encoded1, dim=-1, keepdim=True)
            std2 = torch.std(encoded2, dim=-1, keepdim=True)
            encoded1_normalized = (encoded1 - mean1) / (std1 + 1e-7)
            encoded2_normalized = (encoded2 - mean2) / (std2 + 1e-7)
            return torch.mean(encoded1_normalized * encoded2_normalized, dim=-1)
        else:
            raise ValueError("Error: unkown label name: %s" % label_name)

    def extract(self, df, col_name, as_pandas=True, depth=0, labels=None):
        self.eval()
        data = df[col_name].tolist()
        if labels is None:
            labels = self.label_list
        with torch.no_grad():
            features_list = []
            for i in range(0, len(data), self.batch_size):
                input_sents = data[i:i+self.batch_size]
                input_tensor = self.sents2tensor(input_sents)
                features = self.Encoder(input_tensor)
                if depth == 0:
                    pass
                elif depth >= 1:
                    features = torch.cat([self.extract_hidden_decoder_features(features, label_name, num_layers=depth) for label_name in labels], dim=1)
                features_list += list(features.cpu().detach().numpy())
        if as_pandas == True:
            return pd.DataFrame(features_list)
        else:
            return features_list

    def predict(self, df, as_pandas=True):
        self.eval()
        with torch.no_grad():
            pred_values = {key:[] for key in self.label_list}
            for i in range(0, len(df), self.batch_size):
                rows = df.iloc[i:i+self.batch_size]
                sent1 = rows[self.feature_list[0]].to_list()
                sent2 = rows[self.feature_list[1]].to_list()
                # Concatenate input sentences
                input_sents = sent1 + sent2
                features = self.encode(input_sents)
                for label_name in self.label_list:
                    pred = self.decode(features, label_name, len(rows))
                    pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values, columns=self.label_list)
                return res_df
            else:
                return pred_values

    def evaluate(self, df):
        pred_table = self.predict(df, as_pandas=True)
        results = pd.DataFrame(index=self.label_list, columns=["RMSE","PEARSONR","SPEARMANR"])
        for label_name in self.label_list:
            pred_score = pred_table[label_name].tolist()
            true_score = df[label_name].tolist()
            results.loc[[label_name],['RMSE']] = round(math.sqrt(mean_squared_error(true_score, pred_score)), 6)
            results.loc[[label_name],['PEARSONR']] = round(pearsonr(pred_score, true_score)[0] , 6)
            results.loc[[label_name],['SPEARMANR']] = round(spearmanr(pred_score, true_score)[0] , 6)
        return results

    def fit(self, 
        train_set, 
        val_set,
        calibration_set,
        epochs, 
        learning_rate=1.0e-4, 
        optim_type='AdamW',
        num_warmup_steps=30000,
        T_max=10000, # setup consine LR params, lr = lr_max * 0.5 * (1 + cos( steps / T_max ))
        weight_decay=0.01,
        patience=30
        ):
        # Load the models and optimizer
        optimizers_dict = {
            'AdamW': partial(
                torch.optim.AdamW, 
                lr = learning_rate, 
                betas = (0.9, 0.98),
                weight_decay = weight_decay
            ),
            'Adam': partial(
                torch.optim.Adam,
                lr = learning_rate, 
                betas = (0.9, 0.98),
                weight_decay = weight_decay
            ),
            'SGD': partial(
                torch.optim.SGD,
                lr=learning_rate, 
                momentum=0.8, 
                weight_decay=weight_decay
            ),
            'Adagrad': partial(
                torch.optim.Adagrad,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'Adadelta': partial(
                torch.optim.Adadelta,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'RMSprop': partial(
                torch.optim.RMSprop,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'Adamax': partial(
                torch.optim.Adamax,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'Rprop': partial(
                torch.optim.Rprop,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
        }
        if optim_type not in optimizers_dict:
            print(f"Invalid optimizer type {optim_type}. Defaulting to AdamW.")
            optim_type = 'AdamW'
        # Set up the optimizer
        optimizer = optimizers_dict[optim_type](self.parameters())
        # set up the early stop params
        best_score = self.evaluate(val_set)["SPEARMANR"].mean() * self.evaluate(calibration_set)["SPEARMANR"].mean()
        print(f"NOTE: The initial model score is {round(best_score, 4)}.")
        batch_group = 50
        mini_epoch = 500
        batch_id = 0
        # setup warm up params
        warmup_factor = 0.1
        # Apply warm-up learning rate
        if num_warmup_steps > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * warmup_factor 
        # Set a counter to monitor the network reaching convergence
        counter = 0
        # training
        positive_set = train_set[train_set['ShapeScore']>0.6].reset_index(drop=True)
        negative_set = train_set[train_set['ShapeScore']<=0.6].reset_index(drop=True)
        dataset_size = len(positive_set)
        print(f"NOTE: The positive set (ShapeScore > 0.6) size is {dataset_size}.")
        print(f"NOTE: 2 * {dataset_size} data points per epoch.")
        for epoch in range(epochs):
            self.train()
            pos_subset = positive_set.sample(frac=1).head(dataset_size).reset_index(drop=True)
            neg_subset = negative_set.sample(frac=1).head(dataset_size).reset_index(drop=True)
            start = time.time()
            for i in range(0, dataset_size, self.batch_size//2):
                batch_id += 1
                rows = pd.concat(
                    [
                        pos_subset.iloc[i:i+self.batch_size//2], 
                        neg_subset.iloc[i:i+self.batch_size//2]
                    ], 
                    ignore_index=True
                )
                sent1 = rows[self.feature_list[0]].to_list()
                sent2 = rows[self.feature_list[1]].to_list()
                assert len(sent1) == len(sent2), "The length of sent1 and sent2 must be the same."
                batch_size = len(sent1)
                pred = self.forward(sent1, sent2, batch_size)
                loss = torch.tensor(0.0, requires_grad=True).cuda()
                for label in self.label_list:
                    labels = torch.tensor(rows[label].to_list(), dtype=torch.float32).cuda()
                    label_loss = nn.MSELoss()(pred[label], labels) * self.loss_weight[label]
                    loss = torch.add(loss, label_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_id % batch_group == 0:
                    print(f"Epoch {epoch+1}, batch {batch_id}, time {round(time.time()-start, 2)}, train loss: {loss.item()}")
                    # Set up the learning rate scheduler
                    if batch_id <= num_warmup_steps:
                        # Apply warm-up learning rate schedule
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate * ( warmup_factor + (1 - warmup_factor)* ( batch_id / num_warmup_steps) )
                    elif batch_id > num_warmup_steps:
                        # Apply cosine learning rate schedule
                        lr = learning_rate * ( warmup_factor + (1 - warmup_factor) * np.cos(float(batch_id % T_max)/T_max)) 
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                if batch_id % mini_epoch == 0:
                    # Evaluation on validation set, if provided
                    val_res = self.evaluate(val_set)
                    calibration_res = self.evaluate(calibration_set)
                    self.train()
                    print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the validation set:")
                    print(val_res)
                    print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the calibration set:")
                    print(calibration_res)
                    val_spearmanr = np.mean(val_res["SPEARMANR"].to_list())
                    cal_spearmanr = np.mean(calibration_res["SPEARMANR"].to_list())
                    print(f"NOTE: Epoch {epoch+1}, batch {batch_id}, Val: {round(val_spearmanr, 4)}, Cal: {round(cal_spearmanr, 4)}, patience: {patience-counter}")
                    model_score = val_spearmanr * cal_spearmanr
                    if model_score is None and os.path.exists(f'{self.geminimol_model_name}/GeminiMol.pt'):
                        print("NOTE: The parameters don't converge, back to previous optimal model.")
                        self.load_state_dict(torch.load(f'{self.geminimol_model_name}/GeminiMol.pt'))
                    elif model_score > best_score:
                        if counter > 0:
                            counter -= 1
                        best_score = model_score
                        torch.save(self.state_dict(), f"{self.geminimol_model_name}/GeminiMol.pt")
                    else:
                        counter += 1
                    if counter >= patience:
                        print("NOTE: The parameters was converged, stop training!")
                        break
            if counter >= patience:
                break
        val_res = self.evaluate(val_set)
        print(f"Training over! Evaluating on the validation set:")
        print(val_res)
        print(f"Evaluating on the calibration set:")
        print(calibration_res)
        model_score = np.mean(val_res["SPEARMANR"].to_list()) * np.mean(calibration_res["SPEARMANR"].to_list())
        if model_score > best_score:
            best_score = model_score
            torch.save(self.state_dict(), f"{self.geminimol_model_name}/GeminiMol.pt")
        print(f"Best Model Score: {best_score}")
        self.load_state_dict(torch.load(f'{self.geminimol_model_name}/GeminiMol.pt'))
   
class ImageEncoder(nn.Module):
    def __init__(self,
        encoder_type = 'Qformer_base',
        readout_type = 'MLP',
        readout_activation = 'LeakyReLU',
        num_readout_layers = 2,
        num_out_features = 1024, # 1024
        initialize=False,
        ckpt_path = None,
        use_centroid = [5], # eg. [0,1,2,3,4,5] or None
        ):
        super().__init__()
        self.encoder_type = encoder_type
        self.num_out_features = num_out_features
        # init the CPEncoder
        if self.encoder_type == 'cell_vit':
            self.CPEncoder = cell_encoder(
                                        img_size=512, 
                                        depth=6, 
                                        heads=8, 
                                        mlp_dim=128, 
                                        dropout=0.1, 
                                        hard_pe=False,
                                        use_centroid = use_centroid)
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path)
                pretrained_state_dict = checkpoint['model_state_dict']
                current_state_dict = self.CPEncoder.state_dict()
                pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in current_state_dict}
                current_state_dict.update(pretrained_state_dict)
                self.CPEncoder.load_state_dict(current_state_dict, strict=False)

        elif self.encoder_type == 'Qformer_base':
            # config = get_config_from_str("./phenoscreen/Qformer/cellformer_tiny_frozen_baseline_2048.yaml") # 2048
            config = get_config_from_str("./phenoscreen/Qformer/cellformer_tiny_frozen_baseline.yaml") # 1024
            
            config.defrost()
            config.MODEL.NUM_CLASSES = 28293
            config.freeze()
            self.CPEncoder = build_model(config)
            checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
            msg = self.CPEncoder.load_state_dict(checkpoint['model'], strict=False)
            print(msg)

        self.CPEncoder.to('cuda')

    def forward(self, input_image, positions=None, shapes=None):
        if self.encoder_type != 'cell_vit':
            # print(f'input_image={input_image}')
            encoding = self.CPEncoder(input_image)
        else:
            encoding = self.CPEncoder(input_image, positions, shapes)
        return encoding
    def get_num_layer(self, var_name=""):
        if var_name in ("class_embedding", "positional_embedding", "conv1", "ln_pre"):
            return 0
        elif var_name.startswith("transformer.resblocks"):
            layer_id = int(var_name.split('.')[2])
            return layer_id + 1
        else:
            if self.encoder_type == 'cell_vit':
                return len(self.CPEncoder.transformer.layers)
            elif self.encoder_type == 'Qformer_base':
                return len(self.CPEncoder.layers)
            
class MCP_Matching(nn.Module):  
    def __init__(self, 
            model_name, 
            feature_list = ['smiles1','smiles2'], 
            label_dict = {
                "ShapeScore": 0.2,
                "ShapeAggregation": 0.2,
                "ShapeOverlap": 0.05,
                "ShapeDistance": 0.05,
                "CrossSim": 0.15,
                "CrossAggregation": 0.15,
                "CrossDist": 0.05,
                "CrossOverlap": 0.05,
                "MCS": 0.1
            }, 
            encoding_features = 2048, # 1024
            # fig_size = 320,
            image_encoder_type = 'Qformer_base',
            # image_encoder_activation = 'LeakyReLU',
            # image_encoder_readout_type = 'SkipConnection',
            # image_encoder_readout_layers = 3,
            batch_size = 128, 
            metric_list = ['Cosine', 'Pearson', 'KLDiv', 'Euclidean'],
            ckpt_path = None,
        ):
        super(MCP_Matching, self).__init__()
        torch.set_float32_matmul_precision('high') 
        self.model_name = model_name
        self.batch_size = batch_size
        ## create MolecularEncoder module        
        torch.set_float32_matmul_precision('high') 
        self.geminimol_model_name = model_name
        self.batch_size = batch_size
        self.label_list = list(label_dict.keys())
        self.feature_list = feature_list
        ## set loss weight
        loss_weight = list(label_dict.values())
        sum_weight = np.sum(loss_weight)
        self.loss_weight = {key: value/sum_weight for key, value in zip(self.label_list, loss_weight)}
        ## set up featurizer
        self.atom_featurizer = BaseAtomFeaturizer({
            'atom_type':ConcatFeaturizer([
                atom_type_one_hot, 
                atom_hybridization_one_hot, 
                atom_formal_charge, 
                atom_chiral_tag_one_hot, 
                atom_is_in_ring, 
                atom_is_aromatic
            ])}) 
        self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field='bond_type')
        
        ## create MolecularEncoder
        self.Encoder = MolecularEncoder(
            atom_feat_size=self.atom_featurizer.feat_size(feat_name='atom_type'), 
            bond_feat_size=self.bond_featurizer.feat_size(feat_name='bond_type'), 
            gnn_type='WLN',
            num_layers=4, # MOD:12, M5086:4, M2490:12, M5308:4
            num_features=2048, # MOD:1024, M5086:2048, M2490:1024, M5308:1024
            activation='LeakyReLU',
            readout_type='MeanMLP', # MOD:Weighted, M5086:MeanMLP, M2490:Weighted, M5308:Mixed
            integrate_layer_type=None,# MOD:None, M5086:None, M2490:None, M5308:'SkipConnection'
            integrate_layer_num=None,# MOD:None, M5086:None, M2490:None, M5308:2
            num_out_features = 2048
        )
        # create multiple decoders
        self.Decoder = nn.ModuleDict()
        for label in self.label_list:
            self.Decoder[label] = encoding2score(
                n_embed=2048, 
                dropout_rate=0.0,
                expand_ratio=5,
                activation='LeakyReLU'
                )
        if os.path.exists(self.geminimol_model_name):
            if os.path.exists(f'{self.geminimol_model_name}/GeminiMol.pt'):
                self.load_state_dict(torch.load(f'{self.geminimol_model_name}/GeminiMol.pt'))
        else:
            os.mkdir(self.geminimol_model_name)
        
        # # create ImageEncoder module
        self.metric_list = metric_list
        FeatureLinear = nn.Linear(1024, 2048)
        # Apply Kaiming He initialization to the weights
        nn.init.kaiming_normal_(FeatureLinear.weight, mode='fan_out', nonlinearity='relu')
        # Optionally initialize biases to zero
        nn.init.constant_(FeatureLinear.bias, 0)
        FeatureLinear = FeatureLinear.to('cuda:0')
        self.FeatureLinear = nn.DataParallel(FeatureLinear)
        self.image_encoder_type = image_encoder_type
        self.ImageEncoder = nn.DataParallel(
            ImageEncoder(
                encoder_type = image_encoder_type,
                num_out_features = 1024, # encoding_features
                ckpt_path=ckpt_path
        ))
        self.sigmoid = nn.Sigmoid()
        self.mol_gate = nn.DataParallel(
            nn.Sequential(
                nn.Linear(encoding_features, encoding_features),
                nn.LayerNorm(encoding_features),
                nn.Linear(encoding_features, encoding_features),
                nn.Sigmoid()
            )
        )
        self.mol_gate.cuda()
        # load model or make new model
        if os.path.exists(self.model_name):
            if os.path.exists(f'{self.model_name}/GCMol.pt'):   
                self.load_state_dict(torch.load(f'{self.model_name}/GCMol.pt'))
        else:
            os.mkdir(self.model_name)

    def smiles2tensor(self, input_sents):
        input_tensor = batch([smiles_to_bigraph(smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer).to('cuda') for smiles in input_sents])
        return input_tensor
    
    def npz2tensor(self, path):
        data = np.load(path, allow_pickle=True)
        tensor = self.transform(np.array([data['DNA'], data['ER'], data['RNA'], data['AGP'], data['Mito']]).transpose(1,2,0))
        return (path, tensor)
    
    def image2tensor(self, input_path, parallel=True): # True
        if parallel:
            path_img = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for path in input_path:
                    futures.append(executor.submit(self.npz2tensor, path))
                for future in concurrent.futures.as_completed(futures):
                    path_img[future.result()[0]] = future.result()[1]
            return torch.stack([path_img[path] for path in input_path])
        else:
            input_path_img=[
                np.array([
                    np.load(path, allow_pickle=True)['DNA'], 
                    np.load(path, allow_pickle=True)['ER'], 
                    np.load(path, allow_pickle=True)['RNA'], 
                    np.load(path, allow_pickle=True)['AGP'], 
                    np.load(path, allow_pickle=True)['Mito']
                    ]) 
                for path in input_path 
                ]
            return torch.stack([self.transform(img) for img in input_path_img])

    def mol_encode(self, input_sents):
        # Encode all sentences using the encoder 
        input_tensor = self.smiles2tensor(input_sents).to('cuda') # when encode, change to ([input_sents])
        features = self.Encoder(input_tensor)
        return features

    def forward_css(self, sent1, sent2, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # Concatenate input sentences
        input_sents = sent1 + sent2
        input_tensor = self.smiles2tensor(input_sents).to('cuda')
        # Encode all sentences using the encoder
        features = self.Encoder(input_tensor)
        encoded1 = features[:batch_size]
        encoded2 = features[batch_size:]
        encoded = torch.cat((encoded1, encoded2), dim=1)
        # Calculate specific score between encoded smiles
        pred = {}
        for label in self.label_list:
            pred[label] = self.Decoder[label](encoded)
        return pred

    def decode_css(self, features, label_name, batch_size, depth=0):
        # CSS similarity
        if label_name in self.label_list:
            encoded = torch.cat((features[:batch_size], features[batch_size:]), dim=1)
            return self.Decoder[label_name](encoded)
        # re-shape encoded features
        if depth == 0:
            encoded1 = features[:batch_size]
            encoded2 = features[batch_size:]
        else:
            encoded1 = torch.cat([self.extract_hidden_decoder_features(features[:batch_size], label_name, num_layers=depth) for label_name in self.label_list], dim=1)
            encoded2 = torch.cat([self.extract_hidden_decoder_features(features[batch_size:], label_name, num_layers=depth) for label_name in self.label_list], dim=1)
        # decode vector similarity
        if label_name == 'RMSE':
            return 1 - torch.sqrt(torch.mean((encoded1.view(-1, encoded1.shape[-1]) - encoded2.view(-1, encoded2.shape[-1])) ** 2, dim=-1))
        elif label_name == 'Cosine':
            encoded1_normalized = torch.nn.functional.normalize(encoded1, dim=-1)
            encoded2_normalized = torch.nn.functional.normalize(encoded2, dim=-1)
            return torch.nn.functional.cosine_similarity(encoded1_normalized, encoded2_normalized, dim=-1)
        elif label_name == 'Euclidean':
            return 1 - torch.sqrt(torch.sum((encoded1 - encoded2)**2, dim=-1))
        elif label_name == 'Manhattan':
            return 1 - torch.sum(torch.abs(encoded1 - encoded2), dim=-1)
        elif label_name == 'Minkowski':
            return 1 - torch.norm(encoded1 - encoded2, p=3, dim=-1)
        elif label_name == 'KLDiv':
            encoded1_normalized = torch.nn.functional.softmax(encoded1, dim=-1)
            encoded2_log_normalized = torch.nn.functional.log_softmax(encoded2, dim=-1)
            return 1 - (torch.sum(encoded1_normalized * (torch.log(encoded1_normalized + 1e-7) - encoded2_log_normalized), dim=-1))
        elif label_name == 'Pearson':
            mean1 = torch.mean(encoded1, dim=-1, keepdim=True)
            mean2 = torch.mean(encoded2, dim=-1, keepdim=True)
            std1 = torch.std(encoded1, dim=-1, keepdim=True)
            std2 = torch.std(encoded2, dim=-1, keepdim=True)
            encoded1_normalized = (encoded1 - mean1) / (std1 + 1e-7)
            encoded2_normalized = (encoded2 - mean2) / (std2 + 1e-7)
            return torch.mean(encoded1_normalized * encoded2_normalized, dim=-1)
        else:
            raise ValueError("Error: unkown label name: %s" % label_name)

    def image_encode(self, file_path):
        if self.image_encoder_type == 'cell_vit':
            # Encode all images using the encoder
            try:
                numpy_data = np.load(file_path).astype(np.float32)
            except:
                print(file_path)
                raise ValueError
                pass
            try:
                if numpy_data.shape[0] == 0:
                    # print('Empty data', file_path)
                    raise ValueError('Empty data')
            except:
                print(file_path)
            tensor_data = torch.tensor(numpy_data, dtype=torch.float32)
            # add 1 dim at 0 axis
            tensor_data = torch.unsqueeze(tensor_data, 0).to('cuda')

            # if 'cell_train' in file_path:
            #     parquet_path = file_path.replace('cell_train', 'CDRP_train_shape')[0:-4] + '.parquet'
            # elif 'cell_test' in file_path:
            #     parquet_path = file_path.replace('cell_test', 'CDRP_test_shape')[0:-4] + '.parquet'
            # else:
            #     raise ValueError(f'{file_path} is not a valid path')
            parquet_path = file_path.replace('cell_img_data', 'CDRP_img_data_shape')[0:-4] + '.parquet'
            parquet_data = pd.read_parquet(parquet_path, engine="fastparquet")
            if len(parquet_data) == 0:
                print('Empty data', parquet_path)
                raise ValueError('Empty data')
            position_data = parquet_data[['AreaShape_Center_X', 'AreaShape_Center_Y']].to_numpy().astype(np.float32)
            position_data = torch.tensor(position_data, dtype=torch.float32)
            position_data = torch.unsqueeze(position_data, 0).to('cuda')
            shape_data = parquet_data[[\
                'AreaShape_Center_X', 'AreaShape_Center_Y', \
                'AreaShape_BoundingBoxMinimum_X', 'AreaShape_BoundingBoxMaximum_X', \
                'AreaShape_BoundingBoxMinimum_Y', 'AreaShape_BoundingBoxMaximum_Y', \
                'AreaShape_EquivalentDiameter', 'AreaShape_MajorAxisLength', \
                # 'AreaShape_MaxFeretDiameter', 'AreaShape_MaximumRadius', \
                # 'AreaShape_MeanRadius',	'AreaShape_MedianRadius', \
                # 'AreaShape_MinFeretDiameter', 'AreaShape_MinorAxisLength',\
                # 'AreaShape_Orientation','AreaShape_Perimeter'\
            ]].to_numpy().astype(np.float32) / 512
            shape_data = torch.tensor(shape_data, dtype=torch.float32)
            shape_data = torch.unsqueeze(shape_data, 0).to('cuda')

            features = self.ImageEncoder(tensor_data, position_data, shape_data)
            features = self.sigmoid(features)
        else:
            try:
                numpy_data = np.load(file_path).astype(np.float32)
            except:
                print(file_path)
                raise ValueError
                pass
            try:
                if numpy_data.shape[0] == 0:
                    # print('Empty data', file_path)
                    raise ValueError('Empty data')
            except:
                print(file_path)
            x = 92
            y = 4
            cropped_img = numpy_data[y:y+512, x:x+512, :]

            # convert to tensor
            if cropped_img.max() > 1.0:
                img_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1) / 255.0 * 2 - 1
            else:
                img_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1) * 2 - 1
            assert img_tensor.max() <= 1.0
            # add 1 dim at 0 axis
            tensor_data = torch.unsqueeze(img_tensor, 0).to('cuda')
            features = self.ImageEncoder(tensor_data)
            features = self.FeatureLinear(features)
            features = self.sigmoid(features)
        return features

    def decode(self, ref_features, query_features, label_name = 'Cosine', ref_is_image = False):
        if ref_is_image:
            query_features = self.mol_gate(ref_features) * query_features
        if label_name == 'RMSE':
            return 1 - torch.sqrt(torch.mean((ref_features.view(-1, ref_features.shape[-1]) - query_features.view(-1, query_features.shape[-1])) ** 2, dim=-1))
        elif label_name == 'Cosine':
            encoded1_normalized = torch.nn.functional.normalize(ref_features, dim=-1)
            encoded2_normalized = torch.nn.functional.normalize(query_features, dim=-1)
            return torch.nn.functional.cosine_similarity(encoded1_normalized, encoded2_normalized, dim=-1)
        elif label_name == 'Euclidean':
            return 1 - torch.sqrt(torch.sum((ref_features - query_features)**2, dim=-1))
        elif label_name == 'Manhattan':
            return 1 - torch.sum(torch.abs(ref_features - query_features), dim=-1)
        elif label_name == 'Minkowski':
            return 1 - torch.norm(ref_features - query_features, p=3, dim=-1)
        elif label_name == 'KLDiv':
            encoded1_normalized = torch.nn.functional.softmax(ref_features, dim=-1)
            encoded2_log_normalized = torch.nn.functional.log_softmax(query_features, dim=-1)
            return 1 - (torch.sum(encoded1_normalized * (torch.log(encoded1_normalized + 1e-7) - encoded2_log_normalized), dim=-1))
        elif label_name == 'Pearson':
            mean1 = torch.mean(ref_features, dim=-1, keepdim=True)
            mean2 = torch.mean(query_features, dim=-1, keepdim=True)
            std1 = torch.std(ref_features, dim=-1, keepdim=True)
            std2 = torch.std(query_features, dim=-1, keepdim=True)
            encoded1_normalized = (ref_features - mean1) / (std1 + 1e-7)
            encoded2_normalized = (query_features - mean2) / (std2 + 1e-7)
            return torch.mean(encoded1_normalized * encoded2_normalized, dim=-1)
        else:
            raise ValueError("Error: unkown label name: %s" % label_name)

    def extract(self, df, col_name, as_pandas=True):
        self.eval()
        if col_name in ['smiles', 'SMILES', 'molecule', 'compound', 'Drug']:
            encode = self.mol_encode
        elif col_name in ['image', 'Image', 'IMAGE']: 
            encode = self.image_encode
        else:
            raise ValueError("Error: unkown column name: %s" % col_name)
        with torch.no_grad():
            features_list = []
            for input_sents in df[col_name].tolist():
                features = encode(input_sents)
                features_list += list(features.cpu().detach().numpy())
        if as_pandas == True:
            return pd.DataFrame(features_list)
        else:
            return features_list

    def predict_cp(self, df, as_pandas=True):
        self.eval()
        pred_values = {k:[] for k in self.metric_list}
        with torch.no_grad():
            for batch in df:
                if len(batch) == 5:
                    smile, batch_data, batch_position, batch_shape, batch_label = batch
                    start = time.time()
                    mol_features = self.mol_encode(smile)
                    image_features = self.ImageEncoder(batch_data, batch_position, batch_shape)
                    image_features = self.sigmoid(image_features)
                    for label in self.metric_list:
                        pred = self.decode(
                            image_features, 
                            mol_features, 
                            ref_is_image = True,
                            label_name = label
                        )
                        pred_values[label] += list(pred.cpu().detach().numpy())
                elif len(batch) == 3:
                    smile, batch_data, batch_label = batch
                    start = time.time()
                    mol_features = self.mol_encode(smile)
                    image_features = self.ImageEncoder(batch_data)
                    image_features = self.FeatureLinear(image_features)
                    image_features = self.sigmoid(image_features)
                    for label in self.metric_list:
                        pred = self.decode(
                            image_features, 
                            mol_features, 
                            ref_is_image = True,
                            label_name = label
                        )
                        pred_values[label] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values, columns=self.metric_list)
                return res_df
            else:
                return pred_values

    def predict_css(self, df, as_pandas=True):
        self.eval()
        with torch.no_grad():
            pred_values = {key:[] for key in self.label_list}
            for i in range(0, len(df), self.batch_size):
                rows = df.iloc[i:i+self.batch_size]
                sent1 = rows[self.feature_list[0]].to_list()
                sent2 = rows[self.feature_list[1]].to_list()
                # Concatenate input sentences
                input_sents = sent1 + sent2
                features = self.mol_encode(input_sents)
                for label_name in self.label_list:
                    pred = self.decode_css(features, label_name, len(rows))
                    pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values, columns=self.label_list)
                return res_df
            else:
                return pred_values

    def evaluate_gcmol(
            self, 
            df,
            label_col = 'Label'
        ):
        contrastive_results = pd.DataFrame(
            index=self.metric_list, 
            columns=["ACC", "AUROC", "AUPRC"]
        )
        pred_table = self.predict_cp(df, as_pandas=True)
        true_score = []
        for batch in df:
            if len(batch) == 5:
                _,_,_,_,l = batch
                true_score.extend(list(l))
            if len(batch) == 3:
                _,_,l = batch
                true_score.extend(list(l))
        true_score = np.array(true_score)
        for label in self.metric_list:
            pred_score = np.array(pred_table[label].tolist())
            contrastive_results.loc[[label],['ACC']] = round(accuracy_score(true_score, [round(num) for num in pred_score]), 4)
            contrastive_results.loc[[label],['AUROC']] = round(roc_auc_score(true_score, pred_score), 4)
            contrastive_results.loc[[label],['AUPRC']] = round(average_precision_score(true_score, pred_score), 4)
        return contrastive_results # index: label, columns: stat_metrics

    def evaluate_geminimol(self, df):
        pred_table = self.predict_css(df, as_pandas=True)
        results = pd.DataFrame(index=self.label_list, columns=["RMSE","PEARSONR","SPEARMANR"])
        for label_name in self.label_list:
            pred_score = pred_table[label_name].tolist()
            true_score = df[label_name].tolist()
            results.loc[[label_name],['RMSE']] = round(math.sqrt(mean_squared_error(true_score, pred_score)), 6)
            results.loc[[label_name],['PEARSONR']] = round(pearsonr(pred_score, true_score)[0] , 6)
            results.loc[[label_name],['SPEARMANR']] = round(spearmanr(pred_score, true_score)[0] , 6)
        return results

    def fit(self, 
        train_df, # columns: (smiles, image)
        train_set,
        val_set,
        calibration_set,
        epochs = 10, 
        val_df=None, 
        batch_number_per_epoch=10000,
        learning_rate=1.0e-4, 
        lr_ratio_MolEncoder = 1.0, # GeminiMol learning ratio: 0.05, 0.01, 0.001
        lr_ratio_ImageEncoder = 1.0, # VIT learning ratio: 0.05, 0.01, 0.001
        optim_type='AdamW',
        num_warmup_steps=2000,
        T_max=1000,
        batch_group = 10,
        mini_epoch = 100,
        temperature = 0.7,
        weight_decay = 0.001,
        patience = 30 # 3
        ):
        # load params
        self.loss_InfoNCE = InfoNCE(
            temperature=temperature, 
            reduction='mean', 
            negative_mode='unpaired'
        )
        # Load the models and optimizers
        models = {
            'AdamW': partial(
                torch.optim.AdamW, 
                betas = (0.9, 0.98),
                weight_decay = weight_decay
            ),
            'Adam': partial(
                torch.optim.Adam,
                betas = (0.9, 0.98),
                weight_decay = weight_decay
            ),
            'SGD': partial(
                torch.optim.SGD,
                momentum = 0.8, 
                weight_decay=weight_decay
            ),
            'Adagrad': partial(
                torch.optim.Adagrad,
                weight_decay = weight_decay
            ),
            'Adadelta': partial(
                torch.optim.Adadelta,
                weight_decay = weight_decay
            ),
            'RMSprop': partial(
                torch.optim.RMSprop,
                weight_decay = weight_decay
            ),
            'Adamax': partial(
                torch.optim.Adamax,
                weight_decay = weight_decay
            ),
            'Rprop': partial(
                torch.optim.Rprop,
                weight_decay = weight_decay
            ),
        }
        if optim_type not in models:
            print(f"Invalid optimizer type {optim_type}. Defaulting to AdamW.")
            optim_type = 'AdamW'
        # Set up the optimizer
        # param_names = [name for name,_ in self.ImageEncoder.named_parameters()]
        margin_param = 'transformer.layers.4.0.norm.weight' # freeze first 4 layers
        # freeze params before margin_param
        for name, param in self.ImageEncoder.named_parameters():
            if name == margin_param:
                break
            param.requires_grad = False
        optimizer = models[optim_type]([
                {'params': [p for p in self.ImageEncoder.parameters() if p.requires_grad], 'lr': learning_rate*lr_ratio_ImageEncoder},
                {'params': self.FeatureLinear.parameters(), 'lr': learning_rate*lr_ratio_ImageEncoder},
                {'params': self.Encoder.parameters(), 'lr': learning_rate*lr_ratio_MolEncoder},
               #  {'params': self.Decoder.readout.parameters(), 'lr': learning_rate*lr_ratio_MolEncoder},
            ])
        # set up the early stop params
        val_res = self.evaluate_gcmol(
            val_df, label_col = 'Label'
        )
        print(f"Load/Init GCMol, evaluate on the validation set:")
        print(val_res)

        geminimol_val_res = self.evaluate_geminimol(val_set)
        geminimol_calibration_res = self.evaluate_geminimol(calibration_set)

        print(f"Load/Init GCMol, evaluate on the geminimol_validation set:")
        print(geminimol_val_res)
        print(f"Load/Init GCMol, evaluate on the geminimol_calibration set:")
        print(geminimol_calibration_res)

        best_val_score = val_res["AUROC"].max() + val_res["AUPRC"].max()
        bak_val_score = best_val_score
        batch_id = 0
        # setup warm up params
        num_warmup_steps = num_warmup_steps
        warmup_factor = 0.1
        # Apply warm-up learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * warmup_factor 
        # Set a counter to monitor the network reaching convergence
        counter = 0
        # geminimol data
        positive_set = train_set[train_set['ShapeScore']>0.6].reset_index(drop=True)
        negative_set = train_set[train_set['ShapeScore']<=0.6].reset_index(drop=True)
        dataset_size = len(positive_set)
        print(f"NOTE: The positive set (ShapeScore > 0.6) size is {dataset_size}.")
        print(f"NOTE: 2 * {dataset_size} data points per epoch.")
        # training
        for epoch in range(epochs):
            self.train()
            pos_subset = positive_set.sample(frac=1).head(dataset_size).reset_index(drop=True)
            neg_subset = negative_set.sample(frac=1).head(dataset_size).reset_index(drop=True)
            val_score_list = []
            start = time.time()
            batch_id = 0  # Make sure batch_id is initialized outside the loop
            for batch in train_df:
                if len(batch) == 5:
                    batch_id += 1
                    smile, batch_data, batch_position, batch_shape, batch_label = batch
                    self.train()
                    mol_features = self.mol_encode(smile)
                    image_features = self.ImageEncoder(batch_data, batch_position, batch_shape)
                    mol_features = self.mol_gate(image_features) * mol_features
                    image_features = self.sigmoid(image_features)
                elif len(batch) == 3:
                    batch_id += 1
                    smile, batch_data, batch_label = batch
                    self.train()
                    mol_features = self.mol_encode(smile)
                    image_features = self.ImageEncoder(batch_data)
                    image_features = self.FeatureLinear(image_features)
                    mol_features = self.mol_gate(image_features) * mol_features
                    image_features = self.sigmoid(image_features)

                # Processing labels (Tensor)
                unique_labels = torch.unique(batch_label)
                indices = torch.randperm(len(unique_labels))  # Randomly shuffle indices
                split_index = len(unique_labels) // 2

                positive_labels = unique_labels[indices[:split_index]]
                negative_labels = unique_labels[indices[split_index:]]

                # Select data based on positive and negative labels
                pos_mol_features = mol_features[torch.isin(batch_label, positive_labels)].to('cuda')
                neg_mol_features = mol_features[torch.isin(batch_label, negative_labels)].to('cuda')

                pos_image_features = image_features[torch.isin(batch_label, positive_labels)].to('cuda')
                neg_image_features = image_features[torch.isin(batch_label, negative_labels)].to('cuda')

                min_len = min(len(pos_mol_features), len(neg_mol_features))
                pos_mol_features = pos_mol_features[:min_len]
                neg_mol_features = neg_mol_features[:min_len]
                pos_image_features = pos_image_features[:min_len]
                neg_image_features = neg_image_features[:min_len]

                # Calculate the loss
                NCE_loss = torch.mean(
                    self.loss_InfoNCE(
                        pos_mol_features,
                        pos_image_features,
                        torch.cat((neg_mol_features, neg_image_features), dim=0)
                    )
                )

                # GeminiMol loss
                rows = pd.concat(
                    [
                        pos_subset.sample(n=self.batch_size//2), 
                        neg_subset.sample(n=self.batch_size//2)
                    ], 
                    ignore_index=True
                )
                sent1 = rows[self.feature_list[0]].to_list()
                sent2 = rows[self.feature_list[1]].to_list()
                assert len(sent1) == len(sent2), "The length of sent1 and sent2 must be the same."
                batch_size = len(sent1)
                pred = self.forward_css(sent1, sent2, batch_size)
                gm_loss = torch.tensor(0.0, requires_grad=True).cuda()
                for label in self.label_list:
                    labels = torch.tensor(rows[label].to_list(), dtype=torch.float32).cuda()
                    label_loss = nn.MSELoss()(pred[label], labels) * self.loss_weight[label]
                    gm_loss = torch.add(gm_loss, label_loss)

                loss = NCE_loss * gm_loss
                # loss = (1 + gm_loss) * NCE_loss
                # loss = NCE_loss * (distance + 2 - similarity)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_id % batch_group == 0:
                    print(f"Epoch {epoch+1}, batch {batch_id}, time {time.time()-start}, train loss: {round(loss.item(), 4)}, NCE_loss: {round(NCE_loss.item(), 4)}, gm_loss: {round(gm_loss.item(), 4)}")
                # if batch_id % batch_group == 0:
                    # print(f"Epoch {epoch+1}, batch {batch_id}, time {time.time()-start}, train loss: {round(loss.item(), 4)} (NCE: {round(NCE_loss.item(), 4)} distance: {round(distance.item(), 4)} similarity: {round(similarity.item(), 4)})")
                    # Set up the learning rate scheduler
                    if batch_id <= num_warmup_steps:
                        # Apply warm-up learning rate schedule
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate * (warmup_factor + (1 - warmup_factor)* ( batch_id / num_warmup_steps)) 
                    elif batch_id > num_warmup_steps:
                        # Apply cosine learning rate schedule
                        lr = learning_rate * ( warmup_factor + (1 - warmup_factor) * np.cos(float(batch_id % T_max)/T_max)) 
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                if batch_id % mini_epoch == 0:
                    # Evaluation on validation set, if provided
                    if val_df is not None:
                        val_res = self.evaluate_gcmol( 
                            val_df, label_col = 'Label'
                        )

                        geminimol_val_res = self.evaluate_geminimol(val_set)
                        geminimol_calibration_res = self.evaluate_geminimol(calibration_set)
                        
                        self.train()
                        print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the validation set:")
                        print(val_res)

                        print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the geminimol_validation set:")
                        print(geminimol_val_res)
                        print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the geminimol_calibration set:")
                        print(geminimol_calibration_res)

                        val_score = val_res["AUROC"].max() + val_res["AUPRC"].max()
                        val_score_list.append(val_score)
                        if val_score is None and os.path.exists(f'{self.model_name}/GCMol.pt'):
                            print("NOTE: The parameters don't converge, back to previous optimal model.")
                            self.load_state_dict(torch.load(f'{self.model_name}/GCMol.pt'))
                        elif val_score > best_val_score:
                            if counter > 0:
                                counter -= 1
                            best_val_score = val_score
                            torch.save(self.state_dict(), f"{self.model_name}/GCMol.pt")
                        else:
                            counter += 1
                        if counter >= patience:
                            print("NOTE: The parameters was converged, stop training!")
                            break
            if counter >= patience:
                break
        self.load_state_dict(torch.load(f'{self.model_name}/GCMol.pt'))

class PropDecoder(nn.Module):
    def __init__(self, 
        feature_dim = 1024,
        expand_ratio = 0,
        hidden_dim = 1024,
        num_layers = 3,
        dense_dropout = 0.0,
        dropout_rate = 0.1, 
        rectifier_activation = 'LeakyReLU',
        concentrate_activation = 'ELU',
        dense_activation = 'ELU',
        projection_activation = 'ELU',
        projection_transform = 'Sigmoid',
        linear_projection = True
        ):
        super(PropDecoder, self).__init__()
        activation_dict = {
            'GELU': nn.GELU(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'SiLU': nn.SiLU(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU(),
            'CELU': nn.CELU(),
            'Softplus': nn.Softplus(),
            'Softsign': nn.Softsign(),
            'Hardshrink': nn.Hardshrink(),
            'Hardtanh': nn.Hardtanh(),
            'LogSigmoid': nn.LogSigmoid(),
            'Softshrink': nn.Softshrink(),
            'PReLU': nn.PReLU(),
            'Softmin': nn.Softmin(),
            'Softmax': nn.Softmax(),
            'Sigmoid': nn.Sigmoid(),
            'Identity': nn.Identity()
        }
        if expand_ratio > 0:
            self.features_rectifier = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * expand_ratio, bias=True),
                activation_dict[rectifier_activation],
                nn.BatchNorm1d(feature_dim * expand_ratio),
                nn.Dropout(p=dropout_rate, inplace=False), 
                nn.Linear(feature_dim * expand_ratio, hidden_dim, bias=True),
                activation_dict[rectifier_activation],
                nn.BatchNorm1d(hidden_dim),
            )
        else:
            self.features_rectifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim, bias=True),
                activation_dict[rectifier_activation],
                nn.BatchNorm1d(hidden_dim),
            )
        self.concentrate = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(hidden_dim, 128, bias=True), 
            activation_dict[concentrate_activation],
            nn.BatchNorm1d(128)
        )
        self.dense_layers =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128, bias=True),
                activation_dict[dense_activation],
                nn.BatchNorm1d(128),
                nn.Dropout(p=dense_dropout, inplace=False)
            )
            for _ in range(num_layers)
        ])
        self.projection = nn.Sequential(
            nn.Linear(128, 128, bias=True), 
            activation_dict[projection_activation],
            nn.Linear(128, 1, bias=True),
            activation_dict[projection_transform],
            nn.Linear(1, 1, bias=True) if linear_projection else nn.Identity(),
        )
        self.features_rectifier.cuda()
        self.concentrate.cuda()
        self.dense_layers.cuda()
        self.projection.cuda()
    
    def forward(self, features):
        features = self.features_rectifier(features)
        features = self.concentrate(features)
        for layer in self.dense_layers:
            features = layer(features)
        return self.projection(features).squeeze(-1)

class GCMol(MCP_Matching):
    def __init__(self, 
            model_name, 
            # mol_encoder,
            batch_size=None,
            similarity_metrics_list = ['RMSE', 'Cosine', 'Manhattan', 'Minkowski', 'Euclidean', 'KLDiv', 'Pearson']
        ):
        self.model_name = model_name
        with open(f"{model_name}/model_params.json", 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        if not batch_size is None:
            self.params['batch_size'] = batch_size
        super().__init__(model_name, **self.params)
        self.eval()
        self.similarity_metrics_list = similarity_metrics_list
        if os.path.exists(f'{self.model_name}/propdecoders'):
            for propdecoder_mode in os.listdir(f'{self.model_name}/propdecoders'):
                if os.path.exists(f"{propdecoder_mode}/predictor.pt"):
                    with open(f"{propdecoder_mode}/model_params.json", 'r', encoding='utf-8') as f:
                        params = json.load(f)
                self.Decoder[propdecoder_mode] = PropDecoder(
                        feature_dim = params['feature_dim'],
                        rectifier_type = params['rectifier_type'],
                        rectifier_layers = params['rectifier_layers'],
                        hidden_dim = params['hidden_dim'],
                        dropout_rate = params['dropout_rate'], 
                        expand_ratio = params['expand_ratio'], 
                        activation = params['activation']
                    )
                self.Decoder[propdecoder_mode].load_state_dict(
                    torch.load(f'{propdecoder_mode}/predictor.pt')
                )
    
    def prop_decode(self, 
        df, 
        predictor_name, 
        smiles_name = 'smiles',
        as_pandas = True
        ):
        self.Decoder[predictor_name].eval()
        with torch.no_grad():
            pred_values = []
            for i in range(0, len(df), self.batch_size):
                rows = df.iloc[i:i+self.batch_size]
                smiles = rows[smiles_name].to_list()
                features = self.mol_encode(smiles)
                pred = self.Decoder[predictor_name](features).cuda()
                pred_values += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame({predictor_name: pred_values}, columns=predictor_name)
                return res_df
            else:
                return {predictor_name: pred_values}

    def create_database(self, query_smiles_table, smiles_column='smiles', worker_num=1):
        data = query_smiles_table[smiles_column].tolist()
        query_smiles_table['features'] = None
        with torch.no_grad():
            for i in range(0, len(data), 2*worker_num*self.batch_size):
                input_sents = data[i:i+2*worker_num*self.batch_size]
                input_tensor = self.smiles2tensor(input_sents)
                features = self.Encoder(input_tensor)
                features_list = list(features.cpu().detach().numpy())
                for j in range(len(features_list)):
                    query_smiles_table.at[i+j, 'features'] = features_list[j]
        return query_smiles_table

    def similarity_predict(self, shape_database, ref_smiles_image, positions=None, shapes=None, ref_as_frist=False, as_pandas=True, similarity_metrics=None, worker_num=1):
        if similarity_metrics == None:
            similarity_metrics = self.similarity_metrics_list
        features_list = shape_database['features'].tolist()
        with torch.no_grad():
            pred_values = {key:[] for key in self.similarity_metrics_list}
            if os.path.isfile(ref_smiles_image):
                ref_features = self.image_encode(ref_smiles_image).cuda()
                ref_is_image = True
            else:
                ref_features = self.mol_encode([ref_smiles_image]).cuda()
                ref_is_image = False
            ref_features_np = ref_features.cpu().numpy()
            ref_features_df = pd.DataFrame(ref_features_np)
            for i in range(0, len(features_list), worker_num*self.batch_size):
                features_batch = features_list[i:i+worker_num*self.batch_size]
                query_features = torch.from_numpy(np.array(features_batch)).cuda()
                if ref_as_frist == False:
                    for label_name in self.similarity_metrics_list:
                        pred = self.decode(query_features, ref_features, label_name = label_name, ref_is_image = ref_is_image)
                        pred_values[label_name] += list(pred.cpu().detach().numpy())
                else:
                    for label_name in self.similarity_metrics_list:
                        pred = self.decode(ref_features, query_features, label_name = label_name, ref_is_image = ref_is_image)
                        pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values, columns=self.similarity_metrics_list)
                return res_df
            else:
                return pred_values 

    def virtual_screening(self, 
                          ref_smiles_image_list,
                          query_smiles_table,
                          input_with_features = False,
                          reverse=False,
                          smiles_column='smiles',
                          return_all_col = True,
                          similarity_metrics=None,
                          worker_num=1
                          ):
        # shape_database = self.create_database(query_smiles_table, smiles_column=smiles_column, worker_num=worker_num)
        if input_with_features:
            features_database = query_smiles_table
        else:
            features_database = self.create_database(
                query_smiles_table, 
                smiles_column = smiles_column, 
                worker_num = worker_num
            )
        total_res = pd.DataFrame()
        for ref_smiles_image in ref_smiles_image_list:
            query_scores = self.similarity_predict(
                features_database, 
                ref_smiles_image, 
                ref_as_frist=reverse, 
                as_pandas=True, 
                similarity_metrics=similarity_metrics
            )
            assert len(query_scores) == len(query_smiles_table), f"Error: different length between original dataframe with predicted scores! {ref_smiles_image}"
            # total_res = pd.concat([total_res, query_smiles_table.join(query_scores, how='left')], ignore_index=True)
            if return_all_col:
                total_res = pd.concat([total_res, query_smiles_table.join(query_scores, how='left')], ignore_index=True)
            else:
                total_res = pd.concat([total_res, query_smiles_table[[smiles_column]].join(query_scores, how='left')], ignore_index=True)

        return total_res

    def extract_features(self, query_smiles_table, smiles_column='smiles'):
        shape_features = self.extract(
            query_smiles_table, 
            smiles_column, 
            as_pandas=False
        )
        return pd.DataFrame(shape_features).add_prefix('CPM_')

    def extract_image_features(self, query_table, image_column='image'):
        image_features = self.extract(
            query_table, 
            image_column, 
            as_pandas=False
        )
        return pd.DataFrame(image_features).add_prefix('CPI_')
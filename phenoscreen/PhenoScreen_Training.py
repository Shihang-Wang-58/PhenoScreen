import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import sys
from model.gcmol import MCP_Matching, BinarySimilarity
from model.GeminiMol import GeminiMol
from benchmark import Benchmark
from utils.cell_vit import cell_encoder, TrainDataset, tr_collate_fn, TestDataset, te_collate_fn
from utils.match import match
from utils.Qformer_base_dataset import CellDataset_Train_Base, CellDataset_Test_Base, collate_fn_Qformerbase

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # defalut params
    training_random_seed = 508
    np.random.seed(training_random_seed)
    torch.manual_seed(training_random_seed)
    torch.cuda.manual_seed(training_random_seed) 
    torch.cuda.manual_seed_all(training_random_seed) 
    # read data and build dataset
    data_path = sys.argv[1]
    # set training params
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    encoding_method = sys.argv[4]
    image_encoder_type = sys.argv[5]
    checkpoint_path_to_load = sys.argv[6]
    nce_temperature = float(sys.argv[7])
    model_name = sys.argv[8]
    GeminiMol_data_path = sys.argv[9]
    # set training GeminiMol_params
    GeminiMol_training_random_seed = 1207
    gnn_type = sys.argv[10]
    readout_type = sys.argv[11].split(':')[0]
    num_features = int(sys.argv[11].split(':')[1])
    num_layers = int(sys.argv[11].split(':')[2])
    GeminiMol_encoding_features = int(sys.argv[11].split(':')[3])
    if readout_type in ['Mixed', 'MixedMLP', 'MixedBN', 'AttentiveMLP', 'WeightedMLP', 'MMLP']:
        integrate_layer_type = sys.argv[11].split(':')[4]
        integrate_layer_num = int(sys.argv[11].split(':')[5])
    else:
        integrate_layer_type = 'None'
        integrate_layer_num = 0
    decoder_expand_ratio = int(sys.argv[11].split(':')[6])
    decoder_dropout = float(sys.argv[11].split(':')[7])
    label_dict = {} # ShapeScore:0.2,ShapeAggregation:0.2,ShapeOverlap:0.1,CrossSim:0.2,CrossAggregation:0.1,MCS:0.2
    for label in str(sys.argv[12]).split(','):
        label_dict[label.split(':')[0]] = float(label.split(':')[1])
    geminimol_model_name = sys.argv[13]
    patience = int(sys.argv[14])
    if image_encoder_type == 'cell_vit':
        # read data and build dataset
        train_set = TrainDataset(data_path, os.path.join(data_path, 'train' + '_' + str(training_random_seed) + '.csv'))
        val_set = TestDataset(data_path, os.path.join(data_path, 'val' + '_' + str(training_random_seed) + '.csv'))
        test_set = TestDataset(data_path, os.path.join(data_path, 'test' + '_' + str(training_random_seed) + '.csv'))
        
        # data process
        print('Image-Molecule comparison Val Set: Number=', len(val_set))
        print('Image-Molecule comparison Test Set: Number=', len(test_set))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=tr_collate_fn, num_workers=2)
        validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=te_collate_fn, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=te_collate_fn, num_workers=2)
    elif image_encoder_type == 'Qformer_base':
        train_set = CellDataset_Train_Base(data_path, os.path.join(data_path, 'train' + '_' + str(training_random_seed) + '_all.csv'))
        val_set = CellDataset_Test_Base(data_path, os.path.join(data_path, 'val' + '_' + str(training_random_seed) + '_all.csv'))
        test_set = CellDataset_Test_Base(data_path, os.path.join(data_path, 'test' + '_' + str(training_random_seed) + '_all.csv'))

        # data process
        print('Image-Molecule comparison Val Set: Number=', len(val_set))
        print('Image-Molecule comparison Test Set: Number=', len(test_set))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_Qformerbase, num_workers=2)
        validation_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_Qformerbase, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_Qformerbase, num_workers=2)

    # read data and build dataset
    GeminiMol_train_set = shuffle(pd.read_csv(f'{GeminiMol_data_path}/training.csv'), random_state=training_random_seed)
    GeminiMol_data_set = pd.read_csv(f'{GeminiMol_data_path}/test.csv') 
    GeminiMol_val_set = shuffle(GeminiMol_data_set[GeminiMol_data_set['assign']=="val"], random_state=GeminiMol_training_random_seed)
    GeminiMol_cross_set = shuffle(GeminiMol_data_set[GeminiMol_data_set['assign']=="cross"], random_state=GeminiMol_training_random_seed)
    GeminiMol_test_set = shuffle(GeminiMol_data_set[GeminiMol_data_set['assign']=="test"], random_state=GeminiMol_training_random_seed)
    calibration_set = pd.read_csv(f'{GeminiMol_data_path}/calibration.csv') # ShapeScore >= 0.75 and MCS < 0.4 in training set
    adjacent_set = pd.read_csv(f'{GeminiMol_data_path}/indep_adjacent.csv') # ShapeScore > 0.6 and MCS < 0.4 in val and test set
    # data process
    print('Training Set: Number=', len(train_set))
    print('Validation Set: Number=', len(GeminiMol_val_set))
    print('Test Set: Number=', len(GeminiMol_test_set))
    print('Cross Set: Number=', len(GeminiMol_cross_set))
    del GeminiMol_data_set
   
    # training
    # initial a GraphShape BinarySimilarity model  
    params = {
        "batch_size": batch_size,
        "image_encoder_type": image_encoder_type,
        "metric_list": ['Cosine', 'Pearson', 'KLDiv', 'Euclidean'],
        "ckpt_path" : checkpoint_path_to_load,
    }
    trainer_gcmol = MCP_Matching(
        model_name, 
        # geminimol_model_name,
        # encoder, 
        **params
    )
    # trainer_gcmol = torch.compile(trainer_gcmol)
    with open(f"{model_name}/model_params.json", 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    print("NOTE: the params shown as follow:")
    print(params)
    if epochs > 0:
        print(f"NOTE: Training {model_name} ...")
        print('Training Set: Number=', len(train_set))
        # training 
        trainer_gcmol.fit(
            train_df = train_loader, # columns: (smiles, image)
            val_df = validation_loader, # columns: (smiles, image, Label)
            train_set = GeminiMol_train_set,
            val_set = GeminiMol_val_set,
            calibration_set = calibration_set,
            epochs = epochs, 
            learning_rate = 1.0e-3, # 1.0e-4
            lr_ratio_MolEncoder = 0.01, # GeminiMol learning ratio: 1.0, 0.05, 0.01, 0.001
            lr_ratio_ImageEncoder = 1.0, # ViT learning ratio: 1.0, 0.05, 0.01, 0.001
            optim_type='AdamW',
            num_warmup_steps=5000,
            T_max = 5000,
            batch_number_per_epoch = 10000,
            batch_group = 10, # 10
            mini_epoch = 600, # 200
            temperature = nce_temperature, # 0.7
            weight_decay = 0.0001, # 0.0001, 0.001
            patience = patience # 30, 100
        )
    # test best model
    test_score = trainer_gcmol.evaluate(test_loader, label_col = 'Label')
    print('======== Job Report ========')
    val_score = trainer_gcmol.evaluate(validation_loader, label_col = 'Label')
    print('Model performance on the validation set: ', model_name)
    print(val_score)
    print('Model performance on the testing set: ', model_name)
    print(test_score)
    val_score.to_csv(str(model_name+"/"+model_name+"_valid_results.csv"), index=True, header=True, sep=',')
    test_score.to_csv(str(model_name+"/"+model_name+"_test_results.csv"), index=True, header=True, sep=',')

  
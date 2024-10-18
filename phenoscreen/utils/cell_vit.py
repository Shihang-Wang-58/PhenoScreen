import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from fast_pytorch_kmeans import KMeans
class Normalize01(object):
    def __call__(self, sample):
        # Assuming 'sample' is a NumPy array or tensor
        normalized_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        return normalized_sample
    
class TrainDataset(Dataset):
    def __init__(self, root_dir, map, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.class_names = sorted(os.listdir(root_dir), key=lambda x: int(x))
        self.map = pd.read_csv(map)
        self.data = self._load_data()

    def _load_data(self):
        data = []
        i = 0
        for _, row in self.map.iterrows():
            # print(str(self.map['folder_number']))
            # print(os.path.basename(self.map['image'])[:-4] + '_part_' + str(i) + '.npy')
            file_path = os.path.join(str(row['folder_number']), os.path.basename(row['file_name']))
            data.append((row['smiles'], file_path, int(row['folder_number'])))
            # if i > 1000:
            #     break
            i+=1
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile, file_path, label = self.data[idx]
        file_path = os.path.join(self.root_dir, file_path)
        try:
            numpy_data = np.load(file_path).astype(np.float32)
        except:
            # print(file_path)
            # raise ValueError
            pass
        try:
            if numpy_data.shape[0] == 0:
                # print('Empty data', file_path)
                raise ValueError('Empty data')
        except:
            print(file_path)
        if self.transform:
            numpy_data = self.transform(numpy_data)
        tensor_data = torch.tensor(numpy_data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        parquet_path = file_path.replace('cell_img_data', 'CDRP_img_data_shape')[0:-4] + '.parquet'
        parquet_data = pd.read_parquet(parquet_path, engine="fastparquet")
        if len(parquet_data) == 0:
            print('Empty data', parquet_path)
            raise ValueError('Empty data')
        position_data = parquet_data[['AreaShape_Center_X', 'AreaShape_Center_Y']].to_numpy().astype(np.float32)
        position_data = torch.tensor(position_data, dtype=torch.float32)
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

        return smile, tensor_data, position_data, shape_data, label

def tr_collate_fn(batch):
    # Splitting the batch into data, position_data, and labels
    smile, data, position_data, shape_data, labels = zip(*batch)
    
    # Padding the data and position_data separately
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    position_data = rnn_utils.pad_sequence(position_data, batch_first=True, padding_value=0)
    shape_data = rnn_utils.pad_sequence(shape_data, batch_first=True, padding_value=0)

    return smile, data, position_data, shape_data, torch.tensor(labels)

class TestDataset(Dataset):
    def __init__(self, root_dir, map, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.class_names = sorted(os.listdir(root_dir), key=lambda x: int(x))
        self.map = pd.read_csv(map)
        self.data = self._load_data()

    def _load_data(self):
        data = []
        i = 0
        for _, row in self.map.iterrows():
            # print(str(self.map['folder_number']))
            # print(os.path.basename(self.map['image'])[:-4] + '_part_' + str(i) + '.npy')
            file_path = os.path.join(str(row['folder_number']), os.path.basename(row['file_name']))
            data.append((row['smiles'], file_path, int(row['label'])))
            if i > 10000: # 10000, 100000, 220000: # the number of validation or test set compounds
                break
            i+=1
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile, file_path, label = self.data[idx]
        file_path = os.path.join(self.root_dir, file_path)
        try:
            numpy_data = np.load(file_path).astype(np.float32)
        except:
            # print(file_path)
            # raise ValueError
            pass
        try:
            if numpy_data.shape[0] == 0:
                # print('Empty data', file_path)
                raise ValueError('Empty data')
        except:
            print(file_path)
        if self.transform:
            numpy_data = self.transform(numpy_data)
        tensor_data = torch.tensor(numpy_data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int)
        parquet_path = file_path.replace('cell_img_data', 'CDRP_img_data_shape')[0:-4] + '.parquet'
        parquet_data = pd.read_parquet(parquet_path, engine="fastparquet")
        if len(parquet_data) == 0:
            print('Empty data', parquet_path)
            raise ValueError('Empty data')
        position_data = parquet_data[['AreaShape_Center_X', 'AreaShape_Center_Y']].to_numpy().astype(np.float32)
        position_data = torch.tensor(position_data, dtype=torch.float32)
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

        return smile, tensor_data, position_data, shape_data, label

def te_collate_fn(batch):
    # Splitting the batch into data, position_data, and labels
    smile, data, position_data, shape_data, labels = zip(*batch)
    
    # Padding the data and position_data separately
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    position_data = rnn_utils.pad_sequence(position_data, batch_first=True, padding_value=0)
    shape_data = rnn_utils.pad_sequence(shape_data, batch_first=True, padding_value=0)

    return smile, data, position_data, shape_data, labels
# helpers
torch.backends.cudnn.enabled = False
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # SELU, RELU, GELU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        # self.dropout = dropout

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        # m_r = torch.ones_like(attn) * self.dropout
        # attn = attn + torch.bernoulli(m_r) * -1e12

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., use_centroid = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
            
        self.centroids = []
        self.use_centroid = use_centroid
        if use_centroid == None:
            self.use_centroid = []
            pass
        for i in self.use_centroid:
            centroid_path = f'./phenoscreen/model/centroid/{str(i)}_centroids.pt'
            self.centroids.append(torch.load(centroid_path))

    def forward(self, x):
        depth = 0
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            if depth in self.use_centroid:
                c_idx = self.use_centroid.index(depth)
                kmeans = KMeans(n_clusters=2048)
                kmeans.centroids = self.centroids[c_idx]
                labels = kmeans.predict(x)
                x = kmeans.centroids[labels]

        return self.norm(x)

class cell_encoder(nn.Module):
    def __init__(self, *, img_size, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., hard_pe=False, use_centroid = None):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_dim = 256
        self.shape_dim = 32
        self.emb_dim = 2048
        self.dim = self.img_dim + self.shape_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(dropout)
        self.embimg = nn.Sequential(
            Rearrange('b n c p1 p2 -> b n (p1 p2 c)'),
            nn.LayerNorm(5 * 64 * 64),
            nn.Linear(5 * 64 * 64, self.img_dim),
        )

        self.embshape = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, self.shape_dim),
        )

        self.transformer = Transformer(self.dim, depth, heads, dim_head, mlp_dim, dropout, use_centroid)
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Linear(self.dim, self.emb_dim)
        # self.cosloss = MarginCosineProduct(self.emb_dim, num_classes)
        # self.cosloss = nn.Linear(self.emb_dim, num_classes)
        self.height, self.width = pair(img_size)
        self.div = 4096
        self.pos_emb = nn.Embedding(int(self.height * self.width / self.div), self.dim)
        self.hard_pe = hard_pe

    def forward(self, img, positions, shapes):
        img = img.permute(0, 1, 4, 2, 3)

        img_tokens = self.embimg(img)
        shape_tokens = self.embshape(shapes)
        tokens = torch.cat((img_tokens, shape_tokens), dim=2)
        b, n, _ = tokens.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)  # Replicate cls_token for each item in the batch

        pos_embedding = None
        index_1d = (positions[:, :, 0] * self.width + positions[:, :, 1])
        zeros = torch.zeros((b, 1)).to(index_1d.device)
        index_1d = torch.cat((zeros, index_1d), dim=1) / self.div
        pos_embedding = self.pos_emb(index_1d.to(torch.int))

        x = torch.cat((cls_tokens, tokens), dim=1)
        x = x + pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        
        return self.mlp_head(x)
        # return self.cosloss(self.mlp_head(x))
    
    def absolute_positional_encoding_2d(self, image_height, image_width, positions,encoding_dim):
        new_positions = (positions / torch.tensor([image_height, image_width], dtype=torch.float32).to(self.device) - 0.5) * 2.0

        position_encoding = torch.zeros((positions.shape[0], positions.shape[1] + 1, encoding_dim))
        row, col = new_positions[:, :, 0], new_positions[:, :, 1]
        row_data = torch.cos(2 * np.pi * row)
        col_data = torch.sin(2 * np.pi * col)
        level = 20
        for i in range(encoding_dim):
            if i % 2 == 0:
                position_encoding[:, 1:, i] = row_data ** (i // (encoding_dim // level + 1) + 1)
            else:
                position_encoding[:, 1:, i] = col_data ** (i // (encoding_dim // level + 1) + 1)

        return position_encoding
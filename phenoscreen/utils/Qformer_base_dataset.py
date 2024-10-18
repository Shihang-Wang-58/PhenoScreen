import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
class CellDataset_Train_Base(Dataset):
    def __init__(self, root_dir, mapping_csv):
        self.img_dir = os.path.join(root_dir, 'img_data')
        self.mapping_csv = mapping_csv
        self.map = pd.read_csv(mapping_csv)
        self.data = self._load_data()

    def _load_data(self):
        data = []
        i = 0
        for _, row in self.map.iterrows():
            file_path = os.path.join(str(row['folder_number']), os.path.basename(row['file_name']))
            data.append((row['smiles'], file_path, int(row['folder_number'])))
            i+=1
        return data

    def rotate_image(self, image, angle):
        if angle == 90:
            image = torch.rot90(image, k=1, dims=(-2, -1))
        elif angle == 180:
            image = torch.rot90(image, k=2, dims=(-2, -1))
        elif angle == 270:
            image = torch.rot90(image, k=3, dims=(-2, -1))
        return image

    def __getitem__(self, idx):
        smile, file_path, label = self.data[idx]
        file_path = os.path.join(self.img_dir, file_path)
        try:
            numpy_data = np.load(file_path).astype(np.float32)
        except:
            print(file_path)
            raise ValueError
        if numpy_data.shape[0] == 0:
            print('Empty data', file_path)
            raise ValueError('Empty data')

        # common knowledge: input img size is 520*696, expected img size is 512*512
        # random crop to 512*512
        x = np.random.randint(0, 184)
        y = np.random.randint(0, 8)
        cropped_img = numpy_data[y:y+512, x:x+512, :]

        # convert to tensor
        if cropped_img.max() > 1.0:
            img_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1) / 255.0 * 2 - 1
        else:
            img_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1) * 2 - 1
        assert img_tensor.max() <= 1.0
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        img_tensor = self.rotate_image(img_tensor, angle)
        return smile, img_tensor, torch.tensor(label).int()

    def __len__(self):
        return len(self.data)

class CellDataset_Test_Base(Dataset):
    def __init__(self, root_dir, mapping_csv):
        self.img_dir = os.path.join(root_dir, 'img_data')
        self.mapping_csv = mapping_csv
        self.map = pd.read_csv(mapping_csv)
        self.data = self._load_data()

    def _load_data(self):
        data = []
        i = 0
        for _, row in self.map.iterrows():
            file_path = os.path.join(str(row['folder_number']), os.path.basename(row['file_name']))
            data.append((row['smiles'], file_path, int(row['label'])))
            if i > 50000: # 10000, 50000
                break
            i+=1
        return data

    def __getitem__(self, idx):
        smile, file_path, label = self.data[idx]
        file_path = os.path.join(self.img_dir, file_path)
        try:
            numpy_data = np.load(file_path).astype(np.float32)
        except:
            print(file_path)
            raise ValueError
        if numpy_data.shape[0] == 0:
            print('Empty data', file_path)
            raise ValueError('Empty data')

        # common knowledge: input img size is 520*696, expected img size is 512*512
        # random crop to 512*512
        x = np.random.randint(0, 184)
        y = np.random.randint(0, 8)
        cropped_img = numpy_data[y:y+512, x:x+512, :]

        # convert to tensor
        if cropped_img.max() > 1.0:
            img_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1) / 255.0 * 2 - 1
        else:
            img_tensor = torch.from_numpy(cropped_img).permute(2, 0, 1) * 2 - 1
        assert img_tensor.max() <= 1.0
        return smile, img_tensor, torch.tensor(label).int()

    def __len__(self):
        return len(self.data)

def collate_fn_Qformerbase(batch):
    # Splitting the batch into img, label
    smile, img, label = zip(*batch)
    # Padding
    img = rnn_utils.pad_sequence(img, batch_first=True, padding_value=0)
    return smile, img, torch.tensor(label)
import ast
import torch
from os.path import splitext, join
from os import listdir
import csv
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import random
from torchvision import transforms
from utils import set_seed
set_seed(0)
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset(args, dataset_class):
    """
    return split based on given args
    """
    table = csv.reader(open(args.label_path, encoding='utf-8'))
    train_data, val_data, test_data = [], [], []
    non_existing_path_cnt = 0

    try:
        drop_ratio = args.drop_ratio
    except Exception:
        drop_ratio = 0
    for idx, line in enumerate(table):
        if idx > 0:
            if not os.path.exists(line[0]):
                non_existing_path_cnt += 1
            else:
                if line[-1] == 'train':
                    if random.random() < drop_ratio/100:
                        continue
                    train_data.append(line)
                elif line[-1] == 'val':
                    val_data.append(line)
                elif line[-1] == 'test':
                    test_data.append(line)
                else:
                    raise ValueError('bad category string')
    train_dataset = dataset_class(train_data, args, 'train')
    val_dataset = dataset_class(val_data, args, 'val')
    test_dataset = dataset_class(test_data, args, 'test')
    print('-------------------------------------------')
    print(f'#train samples: {len(train_dataset)}')
    print(f'#val samples: {len(val_dataset)}')
    print(f'#test samples: {len(test_dataset)}')
    if non_existing_path_cnt > 0:
        print(f'Warning: {non_existing_path_cnt} lines have non-existing paths, they are dropped.')
    print('-------------------------------------------')


    return train_dataset, val_dataset, test_dataset


class TrainDataset(Dataset):
    """
    randomly shuffle within each series;
    each series has 12 images, 1 projected, 10 interpolated, and 1 original
    """
    def __init__(self, data, args, mode):
        super(TrainDataset, self).__init__()
        self.args = args
        self.data = data
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        painting = 0
        pth = self.data[i][0]
        images = os.listdir(pth)
        all_images = [f'{j}.png' for j in range(11)]

        for k in images:
            if k.startswith('original'):
                all_images.append(k)
                break

        if self.data[i][2] == 'photo':
            ground_truth = list(range(11, -1, -1))
        else:
            painting = 1
            ground_truth = list(range(12))

        dummy = list(zip(all_images, ground_truth))
        random.shuffle(dummy)
        all_images, ground_truth = zip(*dummy)
        output_images = []
        for i in all_images:
            raw_img = Image.open(os.path.join(pth, i))
            output_image = self.transform(raw_img)
            output_images.append(output_image)
        output_images = torch.stack(output_images)

        out = {
            'rank': torch.tensor(ground_truth, dtype=torch.long),
            'image': output_images,
            'painting': painting
        }
        return out


class TrainDatasetShuffled(Dataset):
    """
    randomly shuffle among different series;
    each series has 12 images from randomly selected series while keeping the relative order;
    then randomly shuffle within each series;
    """
    def __init__(self, data, args, mode):
        super(TrainDatasetShuffled, self).__init__()
        self.args = args
        self.data = data
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if os.path.exists(f'label_files/{mode}_file_pths') and os.path.exists(f'label_files/{mode}_paintings.npy'):
            with open(f'label_files/{mode}_file_pths') as f:
                dummy = f.readlines()
            dummy = [ast.literal_eval(d) for d in dummy]
            self.file_pths = dummy
            self.paintings = list(np.load(f'label_files/{mode}_paintings.npy'))
            self.assigned = True
        else:
            self.assigned = False
            self.painting_series, self.photo_series = self.get_series()
            self.series_index, self.index_within_photo_series, self.index_within_painting_series = self.get_index()

    def __len__(self):
        return len(self.data)

    def get_series(self):
        painting_series, photo_series =[], []
        print('generating series...')
        for d in tqdm(self.data):
            pth = d[0]
            all_images = [os.path.join(pth, f'{idx}.png') for idx in range(11)]
            images = os.listdir(pth)
            for k in images:
                if k.startswith('original'):
                    all_images.append(os.path.join(pth, k))
                    break
            if d[2] == 'photo':
                photo_series.append(list(reversed(all_images)))
            else:
                painting_series.append(all_images)
        return painting_series, photo_series

    def get_index(self):
        print('generating index...')
        random.seed(self.args.seed)
        # TODO: check this dataloader
        # 0 for photo and 1 for painting
        series_index = [(0, i1) for i1 in range(len(self.photo_series))] + [(1, i2) for i2 in range(len(self.painting_series))]
        random.shuffle(series_index)
        series_index = {
            idx_dataset: idx_domain for idx_dataset, idx_domain in enumerate(series_index)
        }
        index_within_photo_series = []
        index_within_painting_series = []
        for _ in range(12):
            cur = list(range(len(self.photo_series)))
            cur1 = list(range(len(self.painting_series)))
            random.shuffle(cur)
            random.shuffle(cur1)
            index_within_photo_series.append(cur)
            index_within_painting_series.append(cur1)
        return series_index, index_within_photo_series, index_within_painting_series

    def __getitem__(self, i):
        ground_truth = list(range(12))
        if self.assigned:
            painting = self.paintings[i]
            all_image_pths = self.file_pths[i]

        else:
            idx = self.series_index[i]
            domain = idx[0]
            painting = domain
            domain_idx = idx[1]
            all_image_pths = []
            if domain == 0:
                for k in range(12):
                    idx_of_series = self.index_within_photo_series[k][domain_idx]
                    all_image_pths.append(self.photo_series[idx_of_series][k])
            else:
                for k in range(12):
                    idx_of_series = self.index_within_painting_series[k][domain_idx]
                    all_image_pths.append(self.painting_series[idx_of_series][k])

        dummy = list(zip(all_image_pths, ground_truth))
        random.shuffle(dummy)
        all_images, ground_truth = zip(*dummy)
        output_images = []
        for ipth in all_images:
            raw_img = Image.open(ipth)
            output_image = self.transform(raw_img)
            output_images.append(output_image)
        output_images = torch.stack(output_images)

        out = {
            'rank': torch.tensor(ground_truth, dtype=torch.long),
            'image': output_images,
            'painting': painting
        }
        return out


class InferenceDataset(Dataset):
    """
    Dataset for inference, loading images from a specified directory.
    """
    def __init__(self, args):
        super(InferenceDataset, self).__init__()
        self.args = args
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.images = sorted(os.listdir(args.infer_path))
        self.images = [f for f in self.images if f.lower().endswith('.jpg') or f.lower().endswith('.png') or f.lower().endswith('.jpeg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_file_name = self.images[i]
        raw_img = Image.open(os.path.join(self.args.infer_path, img_file_name))

        out = {
            'image_file_name': img_file_name,
            'image': self.transform(raw_img)
        }
        return out


class ReferenceDataset(Dataset):
    """
    Dataset for comparing metrics that needs content or style reference images.
    """
    def __init__(self, args):
        super(ReferenceDataset, self).__init__()
        self.args = args
        # TODO: check if this is necessary
        self.transform = self.get_transform()

        self.trans_dir = args.transfered_dir
        print(f'trans_dir: {self.trans_dir}')
        self.trans_images = sorted(listdir(self.trans_dir), key=lambda x: splitext(x)[0])

        self.ref_dir = self.get_reference_dir()
        print(f'ref_dir: {self.ref_dir}')
        self.ref_images = {splitext(f)[0]: f for f in listdir(self.ref_dir)}

    def get_transform(self):
        if self.args.metric in ['gram', 'content']:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        elif self.args.metric in ['lpips', 'l2', 'ssim']:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        return transform

    def get_reference_dir(self):
        if self.args.metric in ['gram']:
            return self.args.style_dir

        elif self.args.metric in ['lpips', 'content', 'l2', 'ssim']:
            return self.args.content_dir

    def __len__(self):
        return len(self.trans_images)

    def __getitem__(self, i):
        trans_img = self.trans_images[i]
        trans_img_pth = join(self.trans_dir, trans_img)
        trans_img_pil = Image.open(trans_img_pth).convert('RGB')

        index = splitext(trans_img)[0]
        ref_img = self.ref_images[index]
        ref_img_pth = join(self.ref_dir, ref_img)
        ref_img_pil = Image.open(ref_img_pth).convert('RGB')

        out = {
            'file_name': index,
            'trans': self.transform(trans_img_pil),
            'ref': self.transform(ref_img_pil),
        }
        return out
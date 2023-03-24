# coding=utf-8
# Version:python 3.7

import scipy.io as io
import h5py
from torch.utils.data import DataLoader, Dataset
import glob
import os
from utils import gettransform, ToTensor, channel_minmax_normalize
from utils import minmax_normalize, crop_center, data_augmentation
from add_noise import *


"""
used to create the dataset.
Args:
    matpath & ext: from matpath to load the expansion of the data named ext
    datapath & ratio: the generated data into the datapath (divided into train and test, the segmentation ratio is ratio)
    patch: the size of each image block of the segmentation is patch * patch
    strides: the step size is strides
    mode: selects whether the matpath is a single mat file split or multiple mat files split
        h5py.File: Data applicable to this loading method: icvl & chongqing & urban
        io.loadmat: Data applicable to this loading method: pavia
    channel_first: used to indicate whether the channel is in the first dimension
    key: indicates the keyword of the mat file
    loader: indicates the reading mode of the mat file
    tag: used to mark this segmentation

Results:
    The run result will directly split all mat files in matpath into path-sized blocks and place them in train+tag and test+tag under datapath respectively according to the ratio
    datapath/train & datapath/test
"""
def createdataset(matpath, datapath, ratio, patch, strides, mode, crop_size=None, channel_first=True, key='rad',
                  loader=h5py.File, tag='', ext='.mat'):

    train_dir = datapath + f'train{tag}/'
    test_dir = datapath + f'test{tag}/'
    searchpath = matpath + f'*{ext}'

    mat_list = glob.glob(searchpath)
    split_point = int(len(mat_list) * ratio)
    train_mat, test_mat = mat_list[:split_point], mat_list[split_point:]
    
    if not os.path.exists(datapath):
        os.mkdir(datapath)
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            os.mkdir(train_dir)
            os.mkdir(test_dir)

            tr_count = 0
            te_count = 0
            if mode == 'single':
                # mat_list = [matpath]
                p_c = 0
            else:
                # mat_list = glob.glob(searchpath)
                split_point = int(len(mat_list) * ratio)
                train_mat, test_mat = mat_list[:split_point], mat_list[split_point:]

            for mat_file in mat_list:
                if key is not None:
                    data = loader(mat_file)[key]
                else:
                    data = loader(mat_file)
                data = np.array(data)
 
                if not channel_first:
                    data = np.transpose(data, [2, 0, 1])

                if crop_size:
                    data = crop_center(data, crop_size[0], crop_size[1])
                    
                data = minmax_normalize(data)
                print(data.shape)
                cn = (data.shape[0] - patch[0]) // strides[0] + 1
                wn = (data.shape[1] - patch[1]) // strides[1] + 1
                hn = (data.shape[2] - patch[2]) // strides[2] + 1
                print('create {} images from {} in {}'.format(
                    cn * wn * hn,
                    mat_file, matpath))
                for c in range(cn):
                    for h in range(wn):
                        for w in range(hn):
                            img = data[strides[0] * c: strides[0] * c + patch[0]:,
                                  strides[1] * h: strides[1] * h + patch[1], strides[2] * w: strides[2] * w + patch[2]]
                            if mode == 'single':
                                if p_c % 10:
                                    io.savemat(train_dir + str(tr_count) + '.mat', {'data': img})
                                    tr_count += 1
                                else:
                                    io.savemat(test_dir + str(te_count) + '.mat', {'data': img})
                                    te_count += 1
                                p_c += 1
                            else:
                                if mat_file in train_mat:
                                    io.savemat(train_dir + str(tr_count) + '.mat', {'data': img})
                                    tr_count += 1
                                else:
                                    io.savemat(test_dir + str(te_count) + '.mat', {'data': img})
                                    te_count += 1
        else:
            print('dataset already exist!')
    print('{} images for training, {} images for testing.'.format(len(glob.glob(train_dir + f'*{ext}')),
                                                                  len(glob.glob(test_dir + f'*{ext}'))))


class ICVLData(Dataset):
    def __init__(self, path, count, transforms=None, train=True):
        self.path = path
        self.count = count
        self.transforms = transforms
        self.tensor = ToTensor()
        self.train = train

    def __getitem__(self, index):
        data = np.array(io.loadmat(self.path + f'{index}.mat')['data'])
        # data = channel_minmax_normalize(data, channel_first=True)
        if self.train:
            data = data_augmentation(data)
        img_gt = self.tensor(data)
        if self.transforms:
            noise_img = self.transforms(data)
        # C H W
        return img_gt, noise_img

    def __len__(self):
        return self.count


"""
used to create the dataset as up.
Args:
    blind & noise: noise information

Return:
    First split the file in matpath, place it in datapath, add noise information to it (blind & noise), inherit torch.utils.data.Dataset to get dataloader and return.
"""
def makedataloader(matpath, datapath, patch, strides, mode, channel_first, key, loader, batch_size=(64, 64), blind='no',
                   noise=30,
                   ratio=0.9, crop_size=None):

    createdataset(matpath, datapath, ratio, patch, strides, mode, crop_size, channel_first, key, loader)

    # noise type
    transform_train = gettransform(blind, noise)
    transform_test = gettransform(blind, noise)

    traindataset = ICVLData(datapath + 'train/', len(glob.glob(datapath + 'train/*.mat')), transforms=transform_train,
                            train=True)
    testdataset = ICVLData(datapath + 'test/', len(glob.glob(datapath + 'test/*.mat')), transforms=transform_test,
                           train=False)
    trainDataloader = DataLoader(dataset=traindataset, batch_size=batch_size[0], shuffle=True, num_workers=2)
    testDataloader = DataLoader(dataset=testdataset, batch_size=batch_size[1], shuffle=False, num_workers=2)
    return trainDataloader, testDataloader

import struct
import numpy as np
import scipy.misc
import h5py
import glob
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as scio
import sys
import pdb
import argparse
import time
import gc

mapping = { 0 :'airplane'  ,
            1 :'automobile',
            2 :'bird' ,
            3 :'cat'   ,
            4 :'deer'  ,
            5 :'dog'    ,
            6 :'frog'   ,
            7 :'horse'       ,
            8 :'ship'      ,
            9 :'truck'     }

def connt_to_binary(tem_path1, per):
    data = np.load(tem_path1, allow_pickle=True)
    data_plus = data[data > 0]
    data_plus = np.sort(data_plus)
    lower_q = np.quantile(data_plus, per, interpolation='lower')
    del data_plus
    gc.collect()
    return (data >= lower_q).astype(np.int8)


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, steps=100, count=None,per=0.25):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.steps = steps
        self.count = count
        self.per = per
        path_converted = os.path.join(root, 'converted')
        if not os.path.exists(path_converted):
            os.mkdir(path_converted)
        if count:
            tem_path = os.path.join(path_converted, 'steps{}_count'.format(steps))
        else:
            tem_path = os.path.join(path_converted, 'steps{}_binary_per{}'.format(steps,per))
        print(tem_path)
        
        if not os.path.exists(tem_path):
            os.mkdir(tem_path)

        
        if self.train:
            tem_path1 = os.path.join(tem_path, 'train.npy')
            tem_path2 = os.path.join(tem_path, 'train_label.npy')

            if (not os.path.exists(tem_path1)) | (not os.path.exists(tem_path2)):
                print('dataset not found => creating...')
                if count:
                    time2 = time.time()
                    pre_process(raw_data_path=root,steps=steps,count=count,threshold=per)
                    time2 = time.time() - time2
                    print('create frame image takes %dh %dmin'%(time2/3600, (time2%3600)/60))
                    self.data = np.load(tem_path1, allow_pickle=True)
                    self.targets = np.load(tem_path2, allow_pickle=True)
                    print('load frame image for train successfully')
                else:
                    count_path = os.path.join(path_converted, 'steps{}_count'.format(steps))
                    count_path1 = os.path.join(count_path, 'train.npy')
                    count_path2 = os.path.join(count_path, 'train_label.npy')

                    if (not os.path.exists(count_path1)) | (not os.path.exists(count_path2)):
                        pre_process(raw_data_path=root,steps=steps,count=count,threshold=per)
                
                    self.data = connt_to_binary(count_path1, per)
                    self.targets = np.load(count_path2, allow_pickle=True)
                    np.save(os.path.join(root, 'converted', 'steps{}_binary_per{}'.format(steps,per), 'train.npy'), self.data)
                    np.save(os.path.join(root, 'converted', 'steps{}_binary_per{}'.format(steps,per), 'train_label.npy'), self.targets)
                    print('load frame image for train successfully')

            else:
                self.data = np.load(tem_path1, allow_pickle=True)
                self.targets = np.load(tem_path2, allow_pickle=True)
                print('load frame image for train successfully')
        else:
            tem_path1 = os.path.join(tem_path, 'test.npy')
            tem_path2 = os.path.join(tem_path, 'test_label.npy')

            if (not os.path.exists(tem_path1)) | (not os.path.exists(tem_path2)):
                print('dataset not found => creating...')
                if count:
                    time2 = time.time()
                    pre_process(raw_data_path=root,steps=steps,count=count,threshold=per)
                    time2 = time.time() - time2
                    print('create frame image takes %dh %dmin'%(time2/3600, (time2%3600)/60))
                    self.data = np.load(tem_path1, allow_pickle=True)
                    self.targets = np.load(tem_path2, allow_pickle=True)
                    print('load frame image for test successfully')
                else:
                    count_path = os.path.join(path_converted, 'steps{}_count'.format(steps))
                    count_path1 = os.path.join(count_path, 'test.npy')
                    count_path2 = os.path.join(count_path, 'test_label.npy')

                    if (not os.path.exists(count_path1)) | (not os.path.exists(count_path2)):
                        pre_process(raw_data_path=root,steps=steps,count=count,threshold=per)
                
                    self.data = connt_to_binary(count_path1, per)

                    self.targets = np.load(count_path2, allow_pickle=True)
                    np.save(os.path.join(root, 'converted', 'steps{}_binary_per{}'.format(steps,per), 'test.npy'), self.data)
                    np.save(os.path.join(root, 'converted', 'steps{}_binary_per{}'.format(steps,per), 'test_label.npy'), self.targets)
                    print('load binary frame image for test successfully')

            else:
                self.data = np.load(tem_path1, allow_pickle=True)
                self.targets = np.load(tem_path2, allow_pickle=True)
                print('load frame image for test successfully')
        


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        target = (self.targets[index]) 
        img = torch.from_numpy(img)
        target = torch.tensor(target)

        return (img, target)

    def __len__(self):
        return len(self.data)



def pre_process(raw_data_path, steps=100, count=True, threshold=None):
    print('loading event data')
    train_data,test_data,train_label,test_label = import_dvscifar10(raw_data_path)
    print('start pre-processing')
    num_train = len(train_data)
    num_test = len(test_data)

    # Init the frame data
    train_frame_data = np.zeros([num_train, 2, 128, 128, steps], dtype=np.int8)
    for index,events in enumerate(train_data):
        if (index + 1) % 100 == 0:
            print("\r\tProcessing train data: {:.2f}% complete\t\t".format((index+1) / 90), end='') 
        p = events[:, 3]
        x = events[:, 1]
        y = events[:, 2]
        ts = events[:, 0]
        step_len = ts[-1] // steps

        p_on = (p==1)
        p_off = (p==0)

        x_on = x[p_on]
        y_on = y[p_on]
        ts_on = ts[p_on]

        x_off = x[p_off]
        y_off = y[p_off]
        ts_off = ts[p_off]

        for j in range(steps):
            ts_range = np.where((ts_on >= step_len * j) & (ts_on < step_len * (j + 1)))
            for x1, y1 in zip(x_on[ts_range], y_on[ts_range]):
                train_frame_data[index, 1, x1, y1, j] += 1
                
            ts_range = np.where((ts_off >= step_len * j) & (ts_off < step_len * (j + 1)))
            for x1, y1 in zip(x_off[ts_range], y_off[ts_range]):
                train_frame_data[index, 0, x1, y1, j] += 1
        del events, p, x, y, ts, p_off, p_on, x_off, x_on, y_off, y_on, ts_off, ts_on 
        gc.collect()
    if count:
        np.save(os.path.join(raw_data_path, 'converted', 'steps{}_count'.format(steps), 'train.npy'), train_frame_data)
        np.save(os.path.join(raw_data_path, 'converted', 'steps{}_count'.format(steps), 'train_label.npy'), train_label)
    else:
        np.save(os.path.join(raw_data_path, 'converted', 'steps{}_binary_th{}'.format(steps,per), 'train.npy'), train_frame_data)
        np.save(os.path.join(raw_data_path, 'converted', 'steps{}_binary_th{}'.format(steps,per), 'train_label.npy'), train_label)
    del train_frame_data, train_label, train_data
    gc.collect()
    

    test_frame_data = np.zeros([num_test, 2, 128, 128, steps], dtype=np.int8)
    for index,events in enumerate(test_data):
        if (index + 1) % 100 == 0:
            print("\r\tProcessing test data: {:.2f}% complete\t\t".format((index+1) / 10), end='')
        p = events[:, 3]
        x = events[:, 1]
        y = events[:, 2]
        ts = events[:, 0]
        step_len = ts[-1] // steps

        p_on = (p==1)
        p_off = (p==0)

        x_on = x[p_on]
        y_on = y[p_on]
        ts_on = ts[p_on]

        x_off = x[p_off]
        y_off = y[p_off]
        ts_off = ts[p_off]

        for j in range(steps):
            ts_range = np.where((ts_on >= step_len * j) & (ts_on < step_len * (j + 1)))
            for x1, y1 in zip(x_on[ts_range], y_on[ts_range]):
                test_frame_data[index, 1, x1, y1, j] += 1
                
            ts_range = np.where((ts_off >= step_len * j) & (ts_off < step_len * (j + 1)))
            for x1, y1 in zip(x_off[ts_range], y_off[ts_range]):
                test_frame_data[index, 0, x1, y1, j] += 1
        del events, p, x, y, ts, p_off, p_on, x_off, x_on, y_off, y_on, ts_off, ts_on
        gc.collect()
    if count:
        np.save(os.path.join(raw_data_path, 'converted', 'steps{}_count'.format(steps), 'test.npy'), test_frame_data)
        np.save(os.path.join(raw_data_path, 'converted', 'steps{}_count'.format(steps), 'test_label.npy'), test_label)
    else:
        np.save(os.path.join(raw_data_path, 'converted', 'steps{}_binary_th{}'.format(steps,per), 'test.npy'), test_frame_data)
        np.save(os.path.join(raw_data_path, 'converted', 'steps{}_binary_th{}'.format(steps,per), 'test_label.npy'), test_label)
    del test_frame_data, test_label,test_data 
    gc.collect()
    
    return



    



def import_dvscifar10(raw_data_path):
    events_path = os.path.join(raw_data_path, 'events')
    if not os.path.exists(events_path):
        os.mkdir(events_path)
    path1 = os.path.join(events_path, 'train_data.npy')
    path2 = os.path.join(events_path, 'test_data.npy')
    path3 = os.path.join(events_path, 'train_label.npy')
    path4 = os.path.join(events_path, 'test_label.npy')
    
    if os.path.exists(path1) & os.path.exists(path2) & os.path.exists(path3) & os.path.exists(path4):
        train_data = np.load(path1, allow_pickle=True)
        test_data = np.load(path2, allow_pickle=True)
        train_label = np.load(path3, allow_pickle=True)
        test_label = np.load(path4, allow_pickle=True)
    else:
        print('event data not found => creating...')
        time1 = time.time()
        train_data, test_data, train_label, test_label = create_events(raw_data_path)
        time1 = time.time() - time1
        print('create event data takes %dh %dmin'%(time1/3600, (time1%3600)/60))
    return train_data, test_data, train_label, test_label




def create_events(raw_data_path):
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    index = np.arange(1000)
    test_index = np.random.choice(index.shape[0],100,replace=False)
    train_index = np.delete(index, test_index)

    print("processing raw training data...")
    key = 1
    for i in range(10):
        current_path = os.path.join(raw_data_path, mapping[i])
        for fn in train_index:
            filename = os.path.join(current_path, "{}.mat".format(fn))
            events = scio.loadmat(filename, verify_compressed_data_integrity=False)['out1'].astype(np.int64)#astype
            train_data.append(events)
            train_label.append(i)
            if key % 100 == 0:
                print("\r\tProcessing train data: {:.2f}% complete\t\t".format(key / 90), end='') 
            key += 1

    print("\nprocessing testing data...")
    key = 1
    for i in range(10):
        current_path = os.path.join(raw_data_path, mapping[i])
        for fn in test_index:
            filename = os.path.join(current_path, "{}".format(fn) + '.mat')
            events = scio.loadmat(filename, verify_compressed_data_integrity=False)['out1'].astype(np.int64)#astype
            test_data.append(events)
            test_label.append(i)
            if key % 100 == 0:
                print("\r\tTest data {:.2f}% complete\t\t".format(key / 10), end='')
            key += 1
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    events_path = os.path.join(raw_data_path, 'events')
    if not os.path.exists(events_path):
        os.mkdir(events_path)
    np.save(os.path.join(events_path, 'train_data.npy'), train_data)
    np.save(os.path.join(events_path, 'test_data.npy'), test_data)
    np.save(os.path.join(events_path, 'train_label.npy'), train_label)
    np.save(os.path.join(events_path, 'test_label.npy'), test_label)
    return train_data, test_data, train_label, test_label


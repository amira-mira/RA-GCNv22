import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset


class NTU(Dataset):
    def __init__(self, type='train', setting='cs', data_shape=(3,300,25,2), transform=None):

        self.maxC, self.maxT, self.maxV, self.maxM = data_shape
        self.transform = transform

        file = './datasets/' + setting + '_' + type + '.txt'
        if not os.path.exists(file):
            raise ValueError('Please generate data first! Using gen_data.py in the main folder.')

        fr = open(file, 'r')
        self.files = fr.readlines()
        fr.close()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx].strip()
        label = file_name.split('/')[-1]
        label = int(label.split('A')[1][:3]) - 1

        data = np.zeros((self.maxC, 300, self.maxV, self.maxM))
        location = np.zeros((2, self.maxT, self.maxV, self.maxM))
        with open(file_name, 'r') as fr:
            frame_num = int(fr.readline())
            for frame in range(frame_num):
                if frame >= self.maxT:
                    break
                person_num = int(fr.readline())
                for person in range(person_num):
                    fr.readline()
                    joint_num = int(fr.readline())
                    for joint in range(joint_num):
                        v = fr.readline().split(' ')
                        if joint < self.maxV and person < self.maxM:
                            data[0,frame,joint,person] = float(v[0])
                            data[1,frame,joint,person] = float(v[1])
                            data[2,frame,joint,person] = float(v[2])
                            location[0,frame,joint,person] = float(v[5])
                            location[1,frame,joint,person] = float(v[6])

        if frame_num <= self.maxT:
            data = data[:,:self.maxT,:,:]
        else:
            s = frame_num // self.maxT
            r = random.randint(0, frame_num - self.maxT * s)
            new_data = np.zeros((self.maxC, self.maxT, self.maxV, self.maxM))
            for i in range(self.maxT):
                new_data[:,i,:,:] = data[:,r+s*i,:,:]
            data = new_data

        if self.transform:
            (data, location) = self.transform((data, location))

        data = torch.from_numpy(data).float()
        location = torch.from_numpy(location).float()
        label = torch.from_numpy(np.array(label)).long()
        return data, location, label, file_name

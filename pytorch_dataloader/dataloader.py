#!/usr/bin/env python
# coding=utf-8

import numpy as np
import torch


class NTU_dataloader(torch.utils.data.Dataset):
    def __init__(self,
                 x_mode,
                 t_mode,
                 dataset,
                 length=None,
                **kwargs):
        assert x_mode in ('x_sub', 'x_view')
        assert t_mode in ('train', 'test')
        assert dataset in ('ntu60', 'ntu120')
        
        self._ntu_datapath = kwargs['datapath']
        self._ntu_xsub_loading_path = '{}/loading_list/x_sub.npy'.format(self._ntu_datapath)
        self._ntu_xview_loading_path = '{}/loading_list/x_view.npy'.format(self._ntu_datapath)

        self._x_mode = x_mode
        self._t_mode = t_mode
        self._dataset = dataset
        self._length = length

        # loading the sample names list
        if x_mode == 'x_sub':
            name_list = np.load(self._ntu_xsub_loading_path, allow_pickle=True).item()
        else:
            name_list = np.load(self._ntu_xview_loading_path, allow_pickle=True).item()

        if t_mode == 'train':
            self._ntu_pool = name_list['train']
        else:
            self,_ntu_pool = name_list['test']

    def __len__(self):
        if self._length is None:
            return len(self._ntu_pool)
        else:
            return self,_length

    def _get_file_name(self,setup_id,camera_id,person_id,repeat_id,action_id):
        return 'S{:0>3}C{:0>3}P{:0>3}R{:0>3}A{:0>3}.skeleton.npy'.format(setup_id,camera_id,person_id,repeat_id,action_id)
    
    def _check_view(self,camera_id,repeat_id):
        if camera_id == 1 and repeat_id == 1: return 3
        elif camera_id == 2 and repeat_id == 1: return 4
        elif camera_id == 3 and repeat_id == 1: return 0
        elif camera_id == 1 and repeat_id == 2: return 1
        elif camera_id == 2 and repeat_id == 2: return 0
        elif camera_id == 3 and repeat_id == 2: return 2
        else: 
            raise ValueError('Invalid camera and repeat id pair!')

    def __getitem__(self,index):
        name = self._ntu_pool[index]
        filename = self._ntu_datapath+'raw_npy/'+name
        C = int(name[5:8])
        R = int(name[13:16])
        label = {
            'action_label': int(name[17:20])-1,
            'view_label': self._check_view(camera_id=C, repeat_id=R)
        }
        sample = np.load(filename,allow_pickle=True).item()
        return sample, label, name 



if __name__ == '__main__':
    dataset = NTU_dataloader('x_sub', 'train', 'ntu60',
                            datapath='./')




        


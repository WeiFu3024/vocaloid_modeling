import os
import json

import numpy as np
import torch
import torchaudio
import random
import soundfile as sf
from utils import traverse_dir

encoding = {
    'sv': '0',
    'teto' : '0',
}

def is_numeric(s):
    try:
        float(s)  # Tries to convert the string to a float
        return True
    except ValueError:
        return False

class Full_Modeling_Vocaloid_AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_root,
        win_len, 
        norm_tensor=None,
        sr=44100,  
        pre_room=0,
        cond_size=None
    ):
        super().__init__()
        # arguments 
        self.sr          = sr
        self.path_root   = path_root
        self.win_len     = win_len
        self.norm_tensor = norm_tensor
        self.pre_room    = pre_room
        self.cond_size   = cond_size
        self.conds = []

        input_x_path = os.path.join(self.path_root, 'x')
        output_y_path = os.path.join(self.path_root, 'y')

        filelist_x = traverse_dir(input_x_path, is_pure=True, is_sort=True)
        filelist_y = traverse_dir(output_y_path, is_pure=True, is_sort=True)
        
        self.inputs = {}
        self.chunks = []
        
        for idx, (fn_x, fn_y) in enumerate(zip(filelist_x, filelist_y)):
            path_x = os.path.join(input_x_path, fn_x)
            path_y = os.path.join(output_y_path, fn_y)
            
            self.wav_x_len, self.sr_x = sf.info(path_x).frames, sf.info(path_x).samplerate
            
            assert sf.info(path_x).frames == sf.info(path_y).frames

            self.inputs[idx] = {
                'x': path_x,
                'y': path_y
            }

            # condition processing 
            # expected x: path/of/directory/x_d1_d2_d3.wav
            # expected y: path/of/directory/y_d1_d2_d3.wav 
            
            if self.win_len is not None:
                last_chunk_start_frame = self.wav_x_len - self.win_len - self.pre_room  + 1
                for offset in range(self.pre_room, last_chunk_start_frame, self.win_len):
                    self.chunks.append({'wav_id': idx, 'offset': offset})
            else:
                self.chunks.append({'wav_id': idx, 'offset': self.pre_room})

            raw_conds = os.path.splitext(os.path.basename(fn_x))[0].split('_')
            valid_check_conds = os.path.splitext(os.path.basename(fn_y))[0].split('_')
            raw_conds = [encoding.get(x, x) if not is_numeric(x) else x for x in raw_conds]
            valid_check_conds = [encoding.get(x, x) if not is_numeric(x) else x for x in valid_check_conds]

            # checking same condition 
            assert len(raw_conds) == len(valid_check_conds)
            for i in range(0, len(raw_conds)):
                assert raw_conds[i] == valid_check_conds[i]
            
            # extract condition value 
            __conds_candidates = []
            for i in range(0, len(raw_conds)):
                __conds_candidates.append(float(raw_conds[i]))
            
            # normalization 
            if self.norm_tensor and self.cond_size:
                for c_idx in range(self.cond_size):
                    __max_cond = float(max(self.norm_tensor[c_idx]))
                    __min_cond = float(min(self.norm_tensor[c_idx]))
                    __conds_candidates[c_idx] = ((__conds_candidates[c_idx] - __min_cond)/(__max_cond - __min_cond)) * 2 - 1 
            
            self.conds.append(__conds_candidates)

        # if self.win_len is not None:
        #     # compute chunks
        #     for wav_id in range(self.inputs.shape[0]):
        #         last_chunk_start_frame = self.inputs[wav_id].shape[-1] - self.win_len - self.pre_room + 1
        #         for offset in range(self.pre_room, last_chunk_start_frame, self.win_len):
        #             self.chunks.append({'wav_id': wav_id, 'offset': offset})
        # else: 
        #     for wav_id in range(self.inputs.shape[0]):
        #         st = self.pre_room
        #         self.chunks.append({'wav_id': wav_id, 'offset': st})

        random.seed(1223)
        random.shuffle(self.chunks)


    def __getitem__(self, idx):
        wav_id = self.chunks[idx]['wav_id']
        offset = self.chunks[idx]['offset']
        
        if self.win_len is not None:
            input_x, _ = torchaudio.load(
                self.inputs[wav_id]['x'],
                frame_offset=offset-self.pre_room,
                num_frames=self.win_len+self.pre_room
            )
            output_y, _ = torchaudio.load(
                self.inputs[wav_id]['y'],
                frame_offset=offset,
                num_frames=self.win_len
            )
        else:
            input_x, _ = torchaudio.load(
                self.inputs[wav_id]['x'],
                frame_offset=offset-self.pre_room,
                num_frames=self.wav_x_len-offset+self.pre_room
            )
            output_y, _ = torchaudio.load(
                self.inputs[wav_id]['y'],
                frame_offset=offset,
                num_frames=self.wav_x_len-offset
            )
            
        # if self.win_len is not None:
        #     input_x = self.inputs[wav_id][:, offset-self.pre_room:offset+self.win_len]
        #     output_y = self.outputs[wav_id][:, offset:offset + self.win_len]
        # else:
        #     input_x = self.inputs[wav_id][:, offset - self.pre_room:]
        #     output_y = self.outputs[wav_id][:, offset:]
        
        cond =  self.conds[wav_id]

        return input_x, output_y, cond
        
    
    def __len__(self):
        return len(self.chunks)
    
class Inference_Vocaloid_AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_x,
        path_y,
        win_len, 
        norm_tensor=None,
        sr=44100,  
        pre_room=0,
        cond_size=None
    ):
        super().__init__()
        # arguments 
        self.sr          = sr
        self.path_x      = path_x
        self.path_y      = path_y
        self.win_len     = win_len
        self.norm_tensor = norm_tensor
        self.pre_room    = pre_room
        self.cond_size   = cond_size
        self.conds = []

        # load wav files 
        self.wav_x, sr_x = sf.read(path_x)
        self.wav_y, sr_y = sf.read(path_y)

        assert sr_x == sr_y
        assert sr == sr_x

        # expand to 2d shape 
        len_x = len(self.wav_x)
        len_y = len(self.wav_y)
        if len(self.wav_x.shape) == 1:
            self.wav_x = self.wav_x[..., None] # (T, 1)
        if len(self.wav_y.shape) == 1:
            self.wav_y = self.wav_y[..., None] # (T, 1)
        
        self.wav_x = self.wav_x[..., 0:1]
        self.wav_y = self.wav_y[..., 0:1]

        self.min_len = min(len_x, len_y)
        # 
        self.buffer_size = win_len
        self.pre_room =  pre_room
        
        if self.buffer_size is not None:
            self.num_segments = (len_x-self.pre_room) // self.buffer_size
        else:
            self.num_segments = 1 

        if self.num_segments <= 0:
            raise ValueError(f'Dataset contains 0 segment, please use smaller pre_room or buffer_size')
        
        raw_conds = os.path.splitext(os.path.basename(path_x))[0].split('_')
        valid_check_conds = os.path.splitext(os.path.basename(path_y))[0].split('_')
        raw_conds = [encoding.get(x, x) if not is_numeric(x) else x for x in raw_conds]
        valid_check_conds = [encoding.get(x, x) if not is_numeric(x) else x for x in valid_check_conds]

        # checking same condition 
        assert len(raw_conds) == len(valid_check_conds)
        for i in range(0, len(raw_conds)):
            assert raw_conds[i] == valid_check_conds[i]
        
        # extract condition value 
        __conds_candidates = []
        for i in range(0, len(raw_conds)):
            __conds_candidates.append(float(raw_conds[i]))
        
        # normalization 
        if self.norm_tensor and self.cond_size:
            for c_idx in range(self.cond_size):
                __max_cond = float(max(self.norm_tensor[c_idx]))
                __min_cond = float(min(self.norm_tensor[c_idx]))
                __conds_candidates[c_idx] = ((__conds_candidates[c_idx] - __min_cond)/(__max_cond - __min_cond)) * 2 - 1 
        
        self.conds = __conds_candidates

    def __getitem__(self, idx):

        if self.buffer_size is not None:

            st = self.pre_room + idx * self.buffer_size
            ed = st + self.buffer_size 

            if st >= self.min_len or ed >= self.min_len:
                raise ValueError(f'Index out of bound.')
            
        else:
            st = self.pre_room
            ed = self.min_len

        input_wav_x = self.wav_x[(st-self.pre_room):ed, 0:1].transpose(1, 0)
        target_wav_y = self.wav_y[st:ed, 0:1].transpose(1, 0)

        if self.cond_size:
            return input_wav_x, target_wav_y, self.conds
        else:
            return input_wav_x, target_wav_y
    
    def __len__(self):
        return self.num_segments
    



class SnapShot_AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        input_data_path, 
        target_data_path, 
        sr, 
        win_len, 
        pre_room
    ): 
        super().__init__()

        # load wav files 
        self.wav_x, sr_x = sf.read(input_data_path)
        self.wav_y, sr_y = sf.read(target_data_path)

        assert sr_x == sr_y
        assert sr == sr_x

        # expand to 2d shape 
        len_x = len(self.wav_x)
        len_y = len(self.wav_y)
        if len(self.wav_x.shape) == 1:
            self.wav_x = self.wav_x[..., None] # (T, 1)
        if len(self.wav_y.shape) == 1:
            self.wav_y = self.wav_y[..., None] # (T, 1)
        
        self.wav_x = self.wav_x[..., 0:1]
        self.wav_y = self.wav_y[..., 0:1]

        self.min_len = min(len_x, len_y)
        # 
        self.buffer_size = win_len
        self.pre_room =  pre_room
        
        if self.buffer_size is not None:
            self.num_segments = (len_x-self.pre_room) // self.buffer_size
        else:
            self.num_segments = 1 

        if self.num_segments <= 0:
            raise ValueError(f'Dataset contains 0 segment, please use smaller pre_room or buffer_size')

    def __getitem__(self, idx):

        if self.buffer_size is not None:

            st = self.pre_room + idx * self.buffer_size
            ed = st + self.buffer_size 

            if st >= self.min_len or ed >= self.min_len:
                raise ValueError(f'Index out of bound.')
            
        else:
            st = self.pre_room
            ed = self.min_len

        input_wav_x = self.wav_x[(st-self.pre_room):ed, 0:1].transpose(1, 0)
        target_wav_y = self.wav_y[st:ed, 0:1].transpose(1, 0)
        
        return input_wav_x, target_wav_y
    
    def __len__(self):
        return self.num_segments
    



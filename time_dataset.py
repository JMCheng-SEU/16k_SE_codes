#!/usr/bin/env python
# coding=utf-8
import datetime
import numpy as np
import torch 
from torch.utils.data import Dataset
import torch.utils.data as tud
import os 
import sys
sys.path.append(os.path.dirname(__file__))

import multiprocessing as mp
import soundfile as sf


def read_and_config_file(wave_list, decode=0):
    processed_list = []

    with open(wave_list) as fid:
        if decode:
            for line in fid:
                tmp = line.strip().split()
                sample = {'inputs': tmp[0]}
                processed_list.append(sample)

        else:
            for line in fid:
                tmp = line.strip().split()
                if len(tmp) == 3:
                    sample = {'inputs': tmp[0], 'labels': tmp[1], 'duration': float(tmp[2])}
                elif len(tmp) == 2:
                    sample = {'inputs': tmp[0], 'labels': tmp[1]}
                processed_list.append(sample)
    return processed_list

def audioread(path):
    data,fs = sf.read(path)
    if len(data.shape) >1:
        data = data[0]
    return data

class DataReader(object):
    def __init__(self, file_name):
        self.file_list = read_and_config_file(file_name, decode=True)
    
    def extract_feature(self, path):
        path = path['inputs']
        utt_id = path.split('/')[-1]
        data = audioread(path).astype(np.float32)
        inputs = np.reshape(data, [1, data.shape[0]])
        return inputs, utt_id, data.shape[0]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])


class Processer(object):
    
    def process(self, path, start_time, segement_length):
         
        # wave_inputs = audioread(path['inputs'])
        wave_s1 = audioread(path['labels'])

        noisy_data, fs = sf.read(path['inputs'])

        # wave_s1 = wave_s1 * 1.5
        # noisy_data = noisy_data * 1.5

        # np.clip(wave_s1, -1, 1)
        # np.clip(noisy_data, -1, 1)

        wave_inputs = noisy_data

        # wave_inputs = noisy_data[:, 0]
        # wave_vad = noisy_data[:, 1]




        if start_time == -1:
            wave_inputs = np.concatenate([wave_inputs, wave_inputs[:segement_length-wave_inputs.shape[0]]])
            wave_s1 = np.concatenate([wave_s1, wave_s1[:segement_length-wave_s1.shape[0]]])
            # wave_vad = np.concatenate([wave_vad, wave_vad[:segement_length-wave_vad.shape[0]]])
        else:
            wave_inputs = wave_inputs[start_time:start_time+segement_length]
            wave_s1 = wave_s1[start_time:start_time+segement_length]
            # wave_vad = wave_vad[start_time:start_time+segement_length]
        
        # I find some sample are not fixed to segement_length,
        # so i padding zero it into segement_length
        if wave_inputs.shape[0] != segement_length:
            padded_inputs = np.zeros(segement_length, dtype=np.float32)
            padded_s1 = np.zeros(segement_length, dtype=np.float32)
            padded_vad = np.zeros(segement_length, dtype=np.float32)
            padded_inputs[:wave_inputs.shape[0]] = wave_inputs
            padded_s1[:wave_s1.shape[0]] = wave_s1
            # padded_vad[:wave_vad.shape[0]] = wave_vad
        else:
            padded_inputs = wave_inputs
            padded_s1 = wave_s1
            # padded_vad = wave_vad

        # return padded_inputs, padded_s1, padded_vad

        return padded_inputs, padded_s1

class TimeDataset(Dataset):

    def __init__(
            self,
            scp_file_name,
            segement_length=6,    ### segement_length=0.33 ###0.1125  ###0.337
            sample_rate=16000,
            processer=Processer(),
            gender2spk=None
        ):
        '''
            scp_file_name: the list include:[input_wave_path, output_wave_path, duration]
            spk_emb_scp: a speaker embedding ark's scp 
            segement_length: to clip data in a fix length segment, default: 4s
            sample_rate: the sample rate of wav, default: 16000
            processer: a processer class to handle wave data 
            gender2spk: a list include gender2spk, default: None
        '''
        self.wav_list = read_and_config_file(scp_file_name)
        self.processer = processer
        mgr = mp.Manager()
        self.index = mgr.list()#[d for b in buckets for d in b]
        self.segement_length = int(segement_length * sample_rate)
        starttime = datetime.datetime.now()
        _dochunk(self.wav_list, self.index, self.segement_length, sample_rate)
        endtime = datetime.datetime.now()
        print('The time cost: ')
        print(endtime - starttime)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_info, start_time = self.index[index]
        inputs, s1 = self.processer.process({'inputs':data_info['inputs'],'labels':data_info['labels']}, start_time, self.segement_length)
        return inputs, s1

def worker(target_list, result_list, start, end, segement_length, sample_rate):
    for item in target_list[start:end]:
        duration = item['duration']
        length = int(duration*sample_rate)
        if length < segement_length:
            sample_index = -1
            if length * 2 < segement_length:
                continue
            result_list.append([item, -1])
        else:
            sample_index = 0
            while sample_index + segement_length < length:
                result_list.append(
                        [item, sample_index])
                sample_index += segement_length

            if sample_index <= length:
                    result_list.append([
                        item,
                        int(length - segement_length),
                ])


def _dochunk(wav_list, index, segement_length, sample_rate, num_threads=8):
        # mutliproccesing
        pc_list = []
        stride = len(wav_list) // num_threads
        if stride < 100:
            p = mp.Process(
                            target=worker,
                            args=(
                                    wav_list,
                                    index,
                                    0,
                                    len(wav_list),
                                    segement_length,
                                    sample_rate
                                )
                        )
            p.start()
            pc_list.append(p)
        else: 
            for idx in range(num_threads):
                if idx == num_threads-1:
                    end = len(wav_list)
                else:
                    end = (idx+1)*stride
                p = mp.Process(
                                target=worker,
                                args=(
                                    wav_list,
                                    index,
                                    idx*stride,
                                    end,
                                    segement_length,
                                    sample_rate
                                )
                            )
                p.start()
                pc_list.append(p)
        for p in pc_list:
            p.join()



def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = np.array([len(inputs), max_t, inputs[0].shape[1]])
    inputs_mat = np.zeros(shape, np.float32)
    for idx, inp in enumerate(inputs):
        inputs_mat[idx, :inp.shape[0],:] = inp
    return inputs_mat

def collate_fn(data):
    inputs, s1 = zip(*data)
    inputs = np.array(inputs, dtype=np.float32)
    s1 = np.array(s1, dtype=np.float32)
    return torch.from_numpy(inputs), torch.from_numpy(s1)

def make_loader(scp_file_name, processer=Processer()):
    dataset = TimeDataset(scp_file_name, processer=processer)
    return dataset

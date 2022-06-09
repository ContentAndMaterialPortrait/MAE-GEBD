import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dataset import Kinetics_GEBD_test
from torch.utils.data import DataLoader
from config import *
import pickle 
import argparse

device = torch.device('cuda')

## segmentation alignment
def shift(lyst, duration, m, n, t=0.3):
    x = [0] + [m * 0.1 * duration] * 15
    y = [0] + [n * 0.1 * duration] * 15
    
    margin = (t + duration * 0.05)
    for i in range(len(lyst)):
        if lyst[i] < margin + x[i]:
            margin += x[i]
            lyst[i] = margin
        else:
            margin = lyst[i]
    
    margin = duration - (t + duration * 0.05)
    for i in range(len(lyst)):
        if lyst[-i-1] > margin - y[i]:
            margin -= y[i]
            lyst[-i-1] = margin
        else:
            margin = lyst[-i-1]
    
    lyst = [i for i in lyst if i > 0]
    if len(lyst) != 0 and lyst[0] < (t + duration * 0.05):
        lyst[0] = (t + duration * 0.05)
    
    return lyst

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver', default='')
    args = parser.parse_args()

    test_dataloader = DataLoader(Kinetics_GEBD_test(), batch_size=BATCH_SIZE, shuffle=False)

    prob_result_list = ['base_models/' + i for i in os.listdir('base_models')]
    print("PROB RESULTS LIST BASE:", prob_result_list)
    
    prob_result_pkl_list = []
    for i, prob_result_file in enumerate(prob_result_list):
        print(i)
        with open(prob_result_file, 'rb') as f: 
            prob_result_pkl_list.append(pickle.load(f))
    len_results = len(prob_result_pkl_list)
    print(len_results)

    max_pooling = nn.MaxPool1d(POOL_SIZE, stride=1, padding=POOL_SIZE // 2)

    test_dict = {}
    for feature, filenames, durations in tqdm(test_dataloader):
        out = torch.zeros([len(feature), FEATURE_LEN]).to(device) # [B, L]
        with torch.no_grad():
            for prob_result_pkl in prob_result_pkl_list:
                out += torch.cat([prob_result_pkl[filename] for filename in filenames]) # [B, L]
            out /= len_results

            out = out.unsqueeze(1)

            peak = (out == max_pooling(out))
            peak[out < THRESHOLD] = False
            peak = peak.squeeze()

            idx = torch.nonzero(peak).cpu().numpy()
            
        durations = durations.numpy()

        boundary_list = [[] for _ in range(len(out))]
        for i, j in idx:
            duration = durations[i]
            first = TIME_UNIT/2
            bias = 0

            if j > 1 and j < FEATURE_LEN - 2:
                bias = (out.squeeze()[i][j+2] + out.squeeze()[i][j+1] - out.squeeze()[i][j-1] - out.squeeze()[i][j-2]) / out.squeeze()[i][j] * TIME_UNIT/2.0
                bias = bias.item()
            elif j == 1 or j == FEATURE_LEN-2:
                bias = (out.squeeze()[i][j+1] - out.squeeze()[i][j-1]) / out.squeeze()[i][j] * TIME_UNIT/2.0
                bias = bias.item()

            if TIME_UNIT*j + bias< duration:
                boundary_list[i].append((TIME_UNIT * j + bias).item())

        for i in range(len(boundary_list)):
            boundary_list[i] = shift(boundary_list[i], durations[i], 0.95, 1.00)

        for i, boundary in enumerate(boundary_list):
            filename = filenames[i]
            test_dict[filename] = boundary
    with open('results/test_ensemble_' + args.ver + '.pkl', 'wb') as f:
        pickle.dump(test_dict, f)
    
    print("TEST ENDS!")

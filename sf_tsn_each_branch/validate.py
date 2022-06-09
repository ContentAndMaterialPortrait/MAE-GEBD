import pickle
import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network import SJNET,JWNET
from validation import validate
from dataset import Kinetics_GEBD_train, Kinetics_GEBD_validation, Kinetics_GEBD_test
from tqdm import tqdm
from config import *
from torch.multiprocessing import Pool, Process, set_start_method, cpu_count
import warnings
import argparse

warnings.filterwarnings("ignore")
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_sf', default='')
    parser.add_argument('--model_tsn', default='')
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    validation_dataloader = DataLoader(Kinetics_GEBD_validation(args.fold), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    network_sf = torch.load(args.model_sf)
    network_tsn = torch.load(args.model_tsn)            

    network_sf.eval()
    network_tsn.eval()

    f1_results = {}
    prec_results = {}
    rec_results = {}
    val_dicts = {}
    val_prob_dict = {}
    k = 3

    for s in SIGMA_LIST:
        val_dict = {}
        gaussian_filter = torch.FloatTensor(
			    [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k, k+1)]
				    ).to(device)
        gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)
        gaussian_filter /= torch.max(gaussian_filter)
        gaussian_filter = gaussian_filter.repeat(1, FEATURE_LEN, 1)
        max_pooling = nn.MaxPool1d(5, stride=1, padding=2)

        for feature, filenames, durations in validation_dataloader:
            feature = feature.to(device)
            with torch.no_grad():
                pred_sf = network_sf(feature)
                pred_tsn = network_tsn(feature)
                pred_sf = torch.sigmoid(pred_sf) # [BATCH_SIZE, FEATURE_LEN]
                pred_tsn = torch.sigmoid(pred_tsn) # [BATCH_SIZE, FEATURE_LEN]
                pred = (pred_sf + pred_tsn) / 2

                if s > 0:
                    out = pred.unsqueeze(-1)
                    eye = torch.eye(FEATURE_LEN).to(device)
                    out = out * eye
                    out = nn.functional.conv1d(out, gaussian_filter, padding=k)
                else:
                    out = pred.unsqueeze(1)

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
                    boundary_list[i].append(TIME_UNIT*j + bias)

                for i in range(len(boundary_list)):
                    boundary_list[i] = shift(boundary_list[i], durations[i], 0.95, 1.0)

            for i, boundary in enumerate(boundary_list):
                filename = filenames[i]
                val_dict[filename] = boundary
                val_prob_dict[filename] = out[i]

        val_dicts[s] = val_dict
        f1, prec, rec = validate(val_dict, args.fold)
        f1_results[s] = f1
        prec_results[s] = prec
        rec_results[s] = rec

    print(f'f1: {f1_results}')
    print(f'precision: {prec_results}')
    print(f'recall: {rec_results}')

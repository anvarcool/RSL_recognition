# import pickle
# import os
# import sys
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# import time
# from torch.utils.data import DataLoader
# import torch
# from tqdm import tqdm
# import gc
# import json
# import random


def parse_json(json_path):    
    with open(json_path, 'r') as f:
            data = f.read()
    jsons = {}
    ind = 1
    l = len(data)
    while ind < l:
        n = 1
        k = 0
        ch = ''
        string_json = ''
        while ch != '[':
            ch = data[ind]
            string_json += ch
            ind += 1
        username = string_json[1:-4]
        while (n != 0) or (k != 0):
            ch = data[ind]
            if ch == '{':
                k += 1
            if ch == '[':
                n += 1
            if ch == '}':
                k -= 1
            if ch == ']':
                n -=1
            string_json += ch
            ind += 1
        jsons[username] = '{' + string_json + '}'
        ind += 2
    return jsons


def json_to_tensors_and_save(filenames_train, filenames_test, json_path):
    jsons = parse_json(json_path)
    #Train
    train_h2 = torch.zeros(len(filenames_train), 3, 21, 256)
    train = torch.zeros(len(filenames_train)3, 21, 256)
    for num, item in enumerate(tqdm(filenames_train)):
        d = json.loads(jsons[item])[f'"{item}"']
        out = torch.zeros(3, 21, len(d))
        out_h2 = torch.zeros(3, 21, len(d))
        for i, frame in enumerate(d):
            if 'hand 2' in frame.keys():
                for j, landmark in enumerate(frame['hand 2']):
                    out_h2[0][j][i] = landmark['x']
                    out_h2[1][j][i] = landmark['y']
                    out_h2[2][j][i] = landmark['z']
            if 'hand 1' in frame.keys():
                for j, landmark in enumerate(frame['hand 1']):
                    out[0][j][i] = landmark['x']
                    out[1][j][i] = landmark['y']
                    out[2][j][i] = landmark['z']
        train_h2[num] = out_h2
        train[num] = out
        gc.collect()
        torch.save(train_h2, r'data\train_h2.pt')
        torch.save(train, r'data\train.pt')
    #Test
    test_h2 = torch.zeros(len(filenames_test), 3, 21, 256)
    test = torch.zeros(len(filenames_test), 3, 21, 256)
    for num, item in enumerate(tqdm(filenames_test)):
        d = json.loads(jsons[item])[f'"{item}"']
        out = torch.zeros(3, 21, len(d))
        out_h2 = torch.zeros(3, 21, len(d))
        for i, frame in enumerate(d):
            if 'hand 2' in frame.keys():
                for j, landmark in enumerate(frame['hand 2']):
                    out_h2[0][j][i] = landmark['x']
                    out_h2[1][j][i] = landmark['y']
                    out_h2[2][j][i] = landmark['z']
            if 'hand 1' in frame.keys():
                for j, landmark in enumerate(frame['hand 1']):
                    out[0][j][i] = landmark['x']
                    out[1][j][i] = landmark['y']
                    out[2][j][i] = landmark['z']
        test_h2[num] = out_h2
        test[num] = out
        gc.collect()
        torch.save(test_h2, r'test_h2.pt')
        torch.save(test, r'test.pt')


def collect_data():
    json_path = r'data\slovo_mediapipe.json'
    annotations = pd.read_csv(r'data\annotations.csv', sep = '\t')
    annotations_train = annotations.query('train & (text != "no_event")')
    annotations_test = annotations.query('~train & (text != "no_event")')
    filenames_train = np.array(annotations_train['attachment_id'])
    filenames_test = np.array(annotations_test['attachment_id'])
    json_to_tensors_and_save(filenames_train, filenames_test, json_path)
    gc.collect()

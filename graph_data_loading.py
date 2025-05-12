import gc
import os
import mediapipe as mp
import cv2
from tqdm import *
import numpy as np
from torch_geometric.data import Data
import torch
from mediapipe_transform import image_to_graph

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

alph = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
cnt_dict = {}
fd_dict = {}
fd_names_dict = {}
data_list = []
for letter in alph:
    print(f'\nStarting {letter}')
    dir = f'./ASL_Data/asl_alphabet_train/asl_alphabet_train/{letter}/'
    letter_cnt = 0
    letter_hands_fd = 0
    fd_names = []
    label = alph.index(letter)
    for fname in tqdm(os.listdir(dir)):
        letter_cnt += 1
        path = ''.join([dir, fname])
        image = cv2.imread(path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data_here = image_to_graph(path, label, hands, detection_confidence=0.3)
        if data_here != -1: 
            letter_hands_fd += 1
            data_list.append(data_here)
        #if letter > 1000: break
        gc.collect()
    fd_names_dict[letter] = fd_names 
    cnt_dict[letter] = letter_cnt
    fd_dict[letter] = letter_hands_fd
    print(letter_hands_fd/letter_cnt)
        
hands.close()
torch.save(data_list, 'valid_ASL_data.pt')
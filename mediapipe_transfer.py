import cv2
import os
import torch
import torch.nn as nn
import mediapipe as mp
import pandas as pd
from collections import defaultdict
from torch_geometric.data import Data
from mediapipe.framework.formats import landmark_pb2
from torch_geometric.nn import GCNConv, global_mean_pool

path = os.path.expanduser('./ASL_Data/asl_alphabet_test/asl_alphabet_test/T_test.jpg')
image = cv2.imread(path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
results = hands.process(image_rgb)

def hand_connections_and_landmarks(results: landmark_pb2.NormalizedLandmarkList) -> pd.DataFrame:
    mp_hands = mp.solutions.hands
    connection_map = defaultdict(list)

    for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
        connection_map[start_idx].append(end_idx)
        connection_map[end_idx].append(start_idx) 

    landmark_rows = []
    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        for i, landmark in enumerate(hand_landmarks.landmark):
            landmark_rows.append({
                'hand_index': hand_idx,
                'landmark_index': i,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'connected_to': connection_map[i] 
            })
    return(pd.DataFrame(landmark_rows))

def hand_graph(df: pd.DataFrame, label: int) -> Data:
    df = df.sort_values('landmark_index')
    x = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float)
    edge_list = []

    for _, row in df.iterrows():
        src = row['landmark_index']
        for dst in row['connected_to']:
            edge_list.append((src, dst)) 

    edge_list += [(dst, src) for (src, dst) in edge_list]
    edge_list = list(set(edge_list))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

class HandGNN(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2, device='cpu'):
        super().__init__()
        self.conv1 = GCNConv(in_channels=3, out_channels=16, cached=True, improved=True)
        self.conv2 = GCNConv(in_channels=16, out_channels=32, cached=True, improved=True)
        self.lin = torch.nn.Linear(32, num_classes)
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.train_loss_history = []
        self.val_loss_history = []

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = nn.ReLU()(x)
        x = self.conv2(x, edge_index)
        x = nn.ReLU()(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)
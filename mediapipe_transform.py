import cv2
import torch
import mediapipe as mp
from collections import defaultdict
from torch_geometric.data import Data

# file: string path to the image
# label: correct label for the image, 0 : A, ..., 25 : Z
# detection_confidence: confidence threshold for hand detection
# Returns a PyTorch Geometric Data object containing the hand landmarks and edges
def image_to_graph(file: str, label: int, hands, detection_confidence=0.3) -> Data:
    # read image
    image = cv2.imread(file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe Hands
    mp_hands = mp.solutions.hands
    #hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=detection_confidence)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks: return -1

    # Create a dictionary for general hand connections
    hand_map = defaultdict(list)
    for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
        hand_map[start_idx].append(end_idx)
        hand_map[end_idx].append(start_idx)

    # Create Data object
    x_list = []
    edge_list = []
    for connection in mp_hands.HAND_CONNECTIONS:
        src, dst = connection
        edge_list.append([src, dst])
        edge_list.append([dst, src]) 
    for hand_landmarks in results.multi_hand_landmarks:
        for i, landmark in enumerate(hand_landmarks.landmark):
            x_list.append([landmark.x, 1 - landmark.y, landmark.z])
            #for connection in hand_map[i]:
            #    edge_list.append((i, connection))
    
    #edge_list += [(j, i) for i, j in edge_list]  # Add reverse edges
    #edge_list = list(set(edge_list))  # Remove duplicates
    x = torch.tensor(x_list, dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    y = torch.tensor(label, dtype=torch.long)

    if edge_index.size(1) > 0:
        max_idx = edge_index.max().item()
        if max_idx >= len(x_list):
            print(f"Warning: Invalid edge index {max_idx} for {len(x_list)} nodes")
            return -1

    #hands.close()
    return Data(x=x, edge_index=edge_index, y=y)

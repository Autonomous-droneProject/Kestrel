import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from dataset import AG_VPReID 
from model import CNNdeepSORT

def extract_features(model, dataloader, device):
    """
    Extracts deep features for all images in the dataloader.
    """
    model.eval()
    features = []
    pids = []
    
    with torch.no_grad():
        for (images, labels) in tqdm(dataloader, desc="Extracting Features", unit="batch"):
            images = images.to(device)
            # Get the embeddings from the model
            embeddings = model(images)
            # Move embeddings to CPU for storage
            features.append(embeddings.detach().cpu())
            pids.append(labels.cpu())
            
    # Concatenate all features and pids from all batches
    features = torch.cat(features, dim=0)
    pids = torch.cat(pids, dim=0)
    
    return features, pids

def calculate_metrics_with_precomputed_ranks(sorted_indices, query_pids, gallery_pids):
    """
    Calculates mAP and Rank-k accuracy from pre-sorted indices.
    Args:
        sorted_indices:   LongTensor [Q, G], each row is the gallery‐index ranking for a query.
        query_pids:       Tensor [Q]
        gallery_pids:     Tensor [G]
    """
    Q = sorted_indices.size(0)
    all_aps = []
    rank_correct = {1: 0, 5: 0, 10: 0}

    for i in range(Q):
        q_pid = query_pids[i]
        # sorted_indices[i] is long[ G ], the indices in gallery_features
        ranked_pids = gallery_pids[sorted_indices[i]]     

        # first‐match
        matches = (ranked_pids == q_pid)
        if not matches.any():
            all_aps.append(0.0)
            continue
        
        # Rank-k
        first_match_idx = torch.nonzero(matches).min().item()
        if first_match_idx < 1:   rank_correct[1] += 1
        if first_match_idx < 5:   rank_correct[5] += 1
        if first_match_idx < 10:  rank_correct[10] += 1

        # AP
        tp = matches.cumsum(dim=0).float()
        precision_at_k = tp / (torch.arange(matches.size(0), device=tp.device) + 1)
        ap = (precision_at_k * matches.float()).sum() / matches.sum().float()
        all_aps.append(ap.item())

    mAP  = float(torch.tensor(all_aps).mean())
    r1   = rank_correct[1]  / Q
    r5   = rank_correct[5]  / Q
    r10  = rank_correct[10] / Q
    return mAP, r1, r5, r10



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a Re-ID model.")
    parser.add_argument('--checkpoint', type=str, default='best_model_checkpoint.pth', help='Path to the model checkpoint file.')
    args = parser.parse_args()

    # --- 1. SETUP ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at '{args.checkpoint}'")
        exit()

    # --- 2. LOAD MODEL ---
    model = CNNdeepSORT(embedding_dim=128).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['loss']:.4f}")

    # --- 3. LOAD AND SPLIT DATA ---
    # Use the same validation transform as in training
    val_transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # We need the original, full dataset to get the validation split
    # This logic should be identical to train.py script
    root = r'B:\Downloads\train' #TODO Make sure this path is correct when running script
    # Important: Load base_dataset WITHOUT transforms to get original images for visualization
    base_dataset_no_transform = AG_VPReID(root_dir=root, transform=None, samples=None)
    base_dataset_with_transform = AG_VPReID(root_dir=root, transform=val_transform, samples=base_dataset_no_transform.samples)

    all_pids_in_base = sorted(list(set([pid for _, pid in base_dataset_no_transform.samples])))
    
    # Use a fixed seed for reproducibility of the split
    random.seed(42)
    val_pids_set = set(random.sample(all_pids_in_base, k=int(0.05 * len(all_pids_in_base))))
        
    # Create query and gallery splits
    pid_to_indices = defaultdict(list)
    for i, (_, pid) in enumerate(base_dataset_no_transform.samples):
        if pid in val_pids_set:
            pid_to_indices[pid].append(i)

    query_indices, gallery_indices = [], []
    for pid, indices in pid_to_indices.items():
        # For each person, one image is query, the rest are gallery
        random.shuffle(indices)
        query_indices.append(indices[0])
        gallery_indices.extend(indices[1:])
    
    # If a person has only one image, they will be in query but not gallery. This is standard.
    query_dataset = Subset(base_dataset_with_transform, query_indices)
    gallery_dataset = Subset(base_dataset_with_transform, gallery_indices)

    print(f"Validation set loaded: {len(query_dataset)} queries, {len(gallery_dataset)} gallery images.")

    query_loader = DataLoader(query_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    gallery_loader = DataLoader(gallery_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # --- 4. EXTRACT FEATURES (on GPU) ---
    query_features, query_pids     = extract_features(model, query_loader, device)
    gallery_features, gallery_pids = extract_features(model, gallery_loader, device)

    # Move them to GPU (if extract_features put them on CPU)
    query_features   = query_features.to(device)
    gallery_features = gallery_features.to(device)
    query_pids       = query_pids.to(device)
    gallery_pids     = gallery_pids.to(device)

    # --- 5. COMPUTE METRICS (VECTORIZED & SORTED) ---
    dist_mat       = torch.cdist(query_features, gallery_features)  # [Q, G], on CUDA
    sorted_indices = dist_mat.argsort(dim=1)                        # [Q, G], on CUDA

    mAP, r1, r5, r10 = calculate_metrics_with_precomputed_ranks(sorted_indices, query_pids, gallery_pids)

    print("\n--- Evaluation Results ---")
    print(f"mAP: {mAP:.2%}")
    print(f"Rank-1: {r1:.2%}, Rank-5: {r5:.2%}, Rank-10: {r10:.2%}")

    # --- 6. VISUALIZE TOP-1 RESULTS (for debugging) ---
    print("\n--- Visualizing Top-1 Matches ---")
    num_to_visualize = 5
    
    for i in range(min(num_to_visualize, len(query_indices))):
        # Get query info
        query_base_idx = query_indices[i]
        query_path, query_pid_val = base_dataset_no_transform.samples[query_base_idx]

        # Get top-ranked gallery info
        top_gallery_idx_in_gallery_subset = sorted_indices[i][0].item()
        top_gallery_base_idx = gallery_indices[top_gallery_idx_in_gallery_subset]
        gallery_path, gallery_pid_val = base_dataset_no_transform.samples[top_gallery_base_idx]

        # Load images using PIL
        query_img = Image.open(query_path)
        gallery_img = Image.open(gallery_path)

        # Plot the images side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        
        ax1.imshow(query_img)
        ax1.set_title(f"Query | PID: {query_pid_val}")
        ax1.axis('off')

        ax2.imshow(gallery_img)
        ax2.set_title(f"Top-1 Gallery Match | PID: {gallery_pid_val}")
        ax2.axis('off')

        fig.suptitle(f'Query #{i+1} | Match Correct: {query_pid_val == gallery_pid_val}')
        plt.show()

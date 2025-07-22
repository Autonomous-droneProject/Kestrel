import os
import sys
import time
import torch
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter #TENSORBOARD
from torch.amp import autocast, GradScaler      # instead of torch.cuda.amp -- deprecated

# Custom Modules
from dataset import AG_VPReID, PKsampler
from model import CNNdeepSORT

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------

def pairwise_euclidean(embeddings):
    # embeddings: [B, D]
    norms = (embeddings**2).sum(1, keepdim=True) # [B, 1], squared L2 norm for each embedding
    d2 = norms - 2*embeddings @ embeddings.t() + norms.t() # [B, B], @: dot product matrix between all pairs, faster than iteration
    return torch.clamp(d2, min=1e-12).sqrt() # [B, B], prevents taking sqrt of negative value

# Best for accuracy and small to medium batch sizes (B<=128) on GPU
def mine_and_compute_triplet_loss(embeddings, labels, margin=0.3):
    distances = pairwise_euclidean(embeddings) # shape: [B, B]
    labels = labels.unsqueeze(1)               # [B, 1]
    mask_positives = labels == labels.t()      # [B, B]
    mask_positives.fill_diagonal_(False)       # prevents the zero-distance between an anchor and itself from being considered a valid positive pair, ensuring a more stable mining process.

    mask_negatives = labels != labels.t()      # [B, B]

    total_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    correct = 0
    triplet_count = 0

    for i in range(embeddings.size(0)):
        d_ap = distances[i][mask_positives[i]]  # distances from anchor to all positives
        d_an = distances[i][mask_negatives[i]]  # distances from anchor to all negatives

        for dp in d_ap:
            semi_hard = d_an[(d_an > dp) & (d_an < dp + margin)]
            if len(semi_hard) == 0:
                continue
            dn = semi_hard[torch.randint(len(semi_hard), (1,)).item()]
            loss = torch.nn.functional.relu(dp - dn + margin)
            total_loss = total_loss + loss
            correct += (dp < dn).float().item()  # track if triplet is "correct"
            triplet_count += 1
    
    avg_loss = total_loss / (triplet_count + 1e-8)
    accuracy = correct / (triplet_count + 1e-8)

    return avg_loss, accuracy, triplet_count

# Vectorized implementation of Batch-All triplet loss (easier to vectorize than semi-hard). Will use if memory tight, or if using very large batches (B>=256)
def batch_all_triplet_loss(embeddings, labels, margin=0.3):
    """
    Computes batch-all triplet loss. It considers all valid triplets and sums their loss.
    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim)
        labels: Tensor of shape (batch_size)
    """
    # 1. Calculate pairwise distance matrix
    # cdist is memory-efficient and computes p-norm distance
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    # 2. Create masks for valid pairs
    anchor_positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    anchor_positive_mask.fill_diagonal_(0) # Exclude anchor-itself pairs

    anchor_negative_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()

    # 3. Reshape for broadcasting
    # We want to compare every d(a, p) with every d(a, n)
    anchor_positive_dist = pairwise_dist.unsqueeze(2) # Shape: [B, B, 1]
    anchor_negative_dist = pairwise_dist.unsqueeze(1) # Shape: [B, 1, B]

    # 4. Compute the loss for all possible triplets
    # The broadcasted shape is [B (anchor), B (negative), B (positive)]
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # 5. Create a mask for valid triplets
    # A triplet (a, p, n) is valid if (a, p) is a positive pair and (a, n) is a negative pair.
    mask = anchor_positive_mask.unsqueeze(2) * anchor_negative_mask.unsqueeze(1)

    # 6. Zero out the loss for invalid triplets and non-violating triplets (loss < 0)
    triplet_loss = torch.nn.functional.relu(triplet_loss)
    triplet_loss = triplet_loss * mask

    # 7. Count the number of positive triplets (where loss > 0)
    num_positive_triplets = torch.sum(triplet_loss > 1e-16).float()

    # 8. Calculate the final loss
    loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    
    # (Optional) Calculate accuracy: % of triplets where d(a,p) < d(a,n)
    correct_triplets = torch.sum((anchor_positive_dist < anchor_negative_dist) * mask).float()
    total_valid_triplets = torch.sum(mask).float()
    accuracy = correct_triplets / (total_valid_triplets + 1e-16)

    return loss, accuracy, num_positive_triplets

def run_epoch(model, dataloader, optimizer, scaler, clip_val, device, is_training):
    # Set model to training mode if in training mode
    model.train() if is_training else model.eval()

    total_loss, total_correct, total_samples = 0., 0., 0
    progress_bar = tqdm(dataloader, f"{'Training' if is_training else 'Evaluating'}...", unit = "batch")

    for (images, labels) in progress_bar:
        # Move to device
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        # Non_blocking lets compute overlap with data transfer (small but free speed-up). Launches host --> GPU copy async if tensor already pinned (it is)

        # Forward pass + mine + loss
        with torch.set_grad_enabled(is_training):
            with autocast(device_type=device):
                embeddings = model(images)
                loss, batch_acc, triplet_count = batch_all_triplet_loss(embeddings, labels)

        if is_training:
            # Backward pass + step and optimization
            scaler.scale(loss).backward()           # 1. Calculate gradients based on the current loss.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val) # Computes total L2 norm for all gradients. If exceeds `clip_val`, rescales them to be equal. Prevents gradient explosion.
            scaler.step(optimizer)                  # 2. Update the model's weights using the calculated gradients. With scaler, undoes the scaling and skips the step if gradients overflowed (inf)
            scaler.update()                         # Adjusts the scale factor up/down depending on whether overflow was detected, keeping training stable.
            optimizer.zero_grad(set_to_none=True)   # 3. Reset gradients to zero for the next iteration. Instead of writing 0s into every grad tensor, it just sets the .grad pointer to None (saving a full CUDA memset each iteration)   
        
        # Metrics
        if triplet_count > 0:
            total_loss += loss.item() * triplet_count # Scale loss by number of triplets it represents
            total_correct += batch_acc * triplet_count # accuracy * count = number correct
        total_samples += triplet_count             # The number of samples is the number of triplets

        # Use total_samples for the progress bar display to avoid division by zero
        current_acc = total_correct / (total_samples + 1e-8)
        progress_bar.set_postfix(loss=loss.item(), acc=current_acc)
    
    avg_loss = total_loss / (total_samples + 1e-8)
    accuracy = total_correct / (total_samples + 1e-8)      # accuracy = % of mined triplets where d(ap) < d(an)
    return avg_loss, accuracy

# ---------------------------------------------------
# 1. SETUP
# ---------------------------------------------------

# This code will only run when you execute `python train.py` directly
if __name__ == '__main__':
    # Hyperparameters
    P_IDENTITIES = 16   # Number of persons per batch
    K_INSTANCES = 8     # Number of images per person

    LEARNING_RATE = 1e-3
    EPOCHS = 50
    BATCH_SIZE = P_IDENTITIES * K_INSTANCES # BATCH_SIZE WILL NOW BE P * K = 128
    # Increased from 64. AMP frees memory, and more data per step = fewer GPU idles

    # Initialize TensorBoard writer with a specific run name
    run_name = f"AG_VPReID_B{BATCH_SIZE}_LR{LEARNING_RATE}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=os.path.join('runs', run_name))  #TENSORBOARD

    # Move the model to the appropriate device (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Tries several convolution algorithms on the first batch, then uses fastest one thereafter.
    # Pays off only if input sizes stay fixed (normalized to 256x128)
    torch.backends.cudnn.benchmark = True 

    scaler = GradScaler() # WIth FP16, gradients can under-flow to 0. This multiplies the loss by a large factor, performs backward, then divides them back down during optimizer.
    clip_val = 1.0 # gradient-clip norm

    # ---------------------------------------------------
    # 2. MODEL, LOSS, OPTIMIZER
    # ---------------------------------------------------

    # Instantiate the model and move to device
    # 751 is the number of classes for Market-1501
    model = CNNdeepSORT(embedding_dim=128).to(device)

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) # AdamW is an upgraded version of Adam
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Decays LR by a factor of 0.1 every 10 epochs
    # Gamma of 0.1 is common -- sharp reduction in LR, allowing model to switch from large adjustments to fine-tuning

    # Load from saved checkpoint if found (most generalized model)
    checkpoint_path = 'best_model_checkpoint.pth'
    start_epoch = 0
    if(len(sys.argv) > 1 and sys.argv[1].lower() == "load"):
        if(os.path.exists(checkpoint_path)):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint.get('scaler_state_dict', {}))
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed at epoch {start_epoch} | val_loss={checkpoint['loss']:.4f}")
        else:
            print("No checkpoint found, starting from scratch.")
    else:
        print("Training from scratch...")
    
    writer.add_text("Run Info", f"Resumed from epoch {start_epoch}" if start_epoch > 0 else "Training from scratch") # TENSORBOARD

    # ---------------------------------------------------
    # 3. DATA LOADING & SMALL VALIDATION SPLIT
    # ---------------------------------------------------

    # Define more robust transforms for training
    # Images are resized and normalized for better model performance.
    train_transform = transforms.Compose([
        transforms.Resize((128, 64)), # native Market-1501
        transforms.RandomHorizontalFlip(p=0.5), # Adds ~1% mAP with no speed cost
        transforms.RandomErasing(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std are pre-calculated mean/stdev of the ImageNet dataset.
        # For each channel of the image: output[channel] = (input[channel] - mean[channel]) / std[channel]
        # Standardizes the pixel value range to train more effectively. Helps converge faster
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 64)), # native Market-1501
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std are pre-calculated mean/stdev of the ImageNet dataset.
        # For each channel of the image: output[channel] = (input[channel] - mean[channel]) / std[channel]
        # Standardizes the pixel value range to train more effectively. Helps converge faster
    ])

    # Load dataset and create DataLoader
    root = r'B:\Downloads\train'

    # 1. Instantiate the dataset ONCE to get all samples and IDs
    print("Loading dataset and building cache if needed...")
    base_dataset = AG_VPReID(root_dir=root, transform=None)
    print(f"Found {len(base_dataset)} images.")

    # 2. Get all unique person IDs from the loaded data
    all_pids = sorted(list(set([pid for _, pid in base_dataset.samples])))

    # 3. Split the PIDs for validation
    val_pids = set(random.sample(all_pids, k=int(0.05 * len(all_pids))))
    train_pids = set(all_pids) - val_pids

    # 4. Create lists of SAMPLES (not indices) for each split
    train_samples = [sample for sample in base_dataset.samples if sample[1] in train_pids]
    val_samples = [sample for sample in base_dataset.samples if sample[1] in val_pids]

    # 5. Create two distinct dataset instances using the filtered samples
    train_ds = AG_VPReID(root_dir=root, transform=train_transform, samples=train_samples)
    val_ds = AG_VPReID(root_dir=root, transform=val_transform, samples=val_samples)

    # Create the sampler
    pk_sampler = PKsampler(train_ds, p=P_IDENTITIES, k=K_INSTANCES)

    # Entire training data
    training_dataloader = DataLoader(
        dataset=train_ds, 
        batch_sampler = pk_sampler,
        #batch_size=BATCH_SIZE, 
        shuffle=False,          # Shuffle must be False when a sampler is provided
        persistent_workers=True,# Keeps DataLoader workers alive across epochs and
        prefetch_factor=4,      # queues 4 batches per worker. Eliminates Windows process-spawn penalty; feeds GPU continuously
        num_workers=4,          # start with 4; 6–8 if CPU has threads to spare
        pin_memory=True)        # pin_memory puts tensors into pinned memory, allowing for faster transfer from CPU to GPU.
                                # Can significantly speed up training. Set to true with GPU.

    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  # no need to shuffle for evaluation
        num_workers=4,
        pin_memory=True, 
        persistent_workers=True)
    
    
    # ---------------------------------------------------
    # 4. TENSORBOARD LOGGING (Initial)
    # ---------------------------------------------------

    print("Logging model graph and sample images to TensorBoard...")
    # Get a single batch from the dataloader to log graph and images
    images, labels = next(iter(training_dataloader))

    grid = make_grid(images[:16], nrow=4, normalize=True)
    writer.add_image('Sample Training Images', grid) #TENSORBOARD
    # Ensure the model and input tensor are on the same device for add_graph
    writer.add_graph(model, images.to(device)) #TENSORBOARD
    print("Done.")

    # ---------------------------------------------------
    # 6. MAIN TRAINING LOOP
    # ---------------------------------------------------
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()

        # Train for one epoch
        avg_loss, accuracy = run_epoch(model, training_dataloader, optimizer, scaler, clip_val, device, is_training=True) #tells you if optimizer is doing its job on the data it sees
        val_loss, val_acc = run_epoch(model, val_loader, optimizer=None, scaler=None, clip_val=None, device=device, is_training=False) #tells you if the network is generalizing; early stopping

        #*NEW* Update the learning rate
        scheduler.step() # Without this, the model is unable to converge.

        # --- TENSORBOARD LOGGING (Per Epoch) ---
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # ----- save best on *validation* loss -----
        #if avg_loss < best_val_loss:
        if val_loss < best_val_loss:
            #best_val_loss = avg_loss
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': val_loss
            }, 'best_model_checkpoint.pth')
            print(f"Epoch {epoch+1}: val_loss improved to {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f" Epoch {epoch+1}: Loss did not improve. Patience = {patience_counter}/{patience}")

        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | Acc: {accuracy:.2%} | Time: {elapsed_time:.2f}s")
        
        # Always save latest weights
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss
        }, 'latest_model_checkpoint_epoch.pth')
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n Early stopping triggered after {epoch+1} epochs.")
            break

    print("\nTraining Finished!")

        
    # ---------------------------------------------------
    # 7. CLEANUP
    # ---------------------------------------------------

    # Close the TensorBoard writer
    writer.close()

    # To view the logs, open a terminal in your project's root directory and run:
    # tensorboard --logdir=runs

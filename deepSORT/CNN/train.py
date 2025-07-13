import os
import sys
import time
import torch
import torch.nn as nn
import torchvision
from datetime import datetime
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler      # instead of torch.cuda.amp -- deprecated


# Custom Modules
import dataset 
from model import CNNdeepSORT

# ---------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------

def training(model, dataloader, loss_fn, optimizer, device):
    model.train() # Set model to training mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    progress_bar = tqdm(dataloader, "Training Progress...", unit = "batch")

    for (image, label) in progress_bar:
        image, label = image.to(device, non_blocking = True), label.to(device, non_blocking = True) 
        # Non_blocking lets compute overlap with data transfer (small but free speed-up). Launches host --> GPU copy async if tensor already pinned (it is)

        # Forward pass
        with autocast(device_type=device):
            prediction = model(image) #Forward propagation
            loss = loss_fn(prediction, label) #Find out how wrong we were
        
        # Backward pass and optimization
        scaler.scale(loss).backward()       # 1. Calculate gradients based on the current loss.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val) # Compues total L2 norm for all gradients. If exceeds `clip_val`, rescales them to be equal. Prevents gradient explosion.
        scaler.step(optimizer)              # 2. Update the model's weights using the calculated gradients. With scaler, undoes the scaling and skips the step if gradients overflowed (inf)
        scaler.update()                     # Adjusts the scale factor up/down depending on whether overflow was detected, keeping training stable.
        optimizer.zero_grad(set_to_none=True)# 3. Reset gradients to zero for the next iteration. Instead of writing 0s into every grad tensor, it just sets the .grad pointer to None (saving a full CUDA memset each iteration)
        
        # Metrics
        total_loss += loss.item()
        total_correct += (prediction.argmax(1) == label).sum().item()
        total_samples += label.size(0)

        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# ---------------------------------------------------
# 1. SETUP
# ---------------------------------------------------

# This code will only run when you execute `python train.py` directly
if __name__ == '__main__':
    # Hyperparameters
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    BATCH_SIZE = 128 # e.g. 32, 64, 128
    # Increased from 64. AMP frees memory, and more data per step = fewer GPU idles

    # Initialize TensorBoard writer with a specific run name
    run_name = f"market1501_B{BATCH_SIZE}_LR{LEARNING_RATE}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=os.path.join('runs', run_name))  

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
    model = CNNdeepSORT(embedding_dim=128, num_classes=751).to(device)

    # Define loss function, optimizer, and learning rate scheduler
    loss_function = nn.CrossEntropyLoss() # Cross Entropy Loss is simpler, so more suitable for validation
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
    
    writer.add_text("Run Info", f"Resumed from epoch {start_epoch}" if start_epoch > 0 else "Training from scratch")

    # ---------------------------------------------------
    # 3. DATA LOADING & SMALL VALIDATION SPLIT
    # ---------------------------------------------------

    # Define more robust transforms for training
    # Images are resized and normalized for better model performance.
    transform = transforms.Compose([
        transforms.Resize((128, 64)), # native Market-1501
        transforms.RandomHorizontalFlip(p=0.5), # Adds ~1% mAP with no speed cost
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std are pre-calculated mean/stdev of the ImageNet dataset.
        # For each channel of the image: output[channel] = (input[channel] - mean[channel]) / std[channel]
        # Standardizes the pixel value range to train more effectively. Helps converge faster
    ])

    # Load dataset and create DataLoader
    Market1501_train_file_path = r'C:\Users\david\Downloads\Market-1501-v15.09.15\bounding_box_train'
    train_data = dataset.Market1501(Market1501_train_file_path, transform=transform)
    
    # Only used for early-stopping / LR checks, NOT for final testing. Tiny split keeps epochs fast but still shows over-/under-fitting trends
    val_len = int(0.05 * len(train_data))   # split 5 % off for quick val
    train_len = len(train_data) - val_len

    # split dataset before building both dataLoaders
    train_ds, val_ds = random_split(train_data, [train_len, val_len]) 

    # Entire training data
    training_dataloader = DataLoader(
        dataset=train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
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
    
    # Evaluation helper
    @torch.no_grad() # we don't track gradients here
    def evaluate(model, loader):
        """
        Runs one full pass on `loader` with AMP and returns:
        (mean_cross_entropy_loss, top-1_accuracy)
        """
        model.eval()                     # turn off dropout / batch-norm update
        total_loss = 0.0
        correct    = 0
        samples    = 0

        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            with autocast(device_type=device):             # AMP even in eval -> less VRAM, faster
                logits = model(imgs)
                loss   = loss_function(logits, lbls)

            total_loss += loss.item()
            correct    += (logits.argmax(1) == lbls).sum().item()
            samples    += lbls.size(0)

        mean_loss = total_loss / len(loader)
        accuracy  = correct / samples
        model.train()                    # restore training mode for caller
        return mean_loss, accuracy
    
    # ---------------------------------------------------
    # 4. TENSORBOARD LOGGING (Initial)
    # ---------------------------------------------------

    print("Logging model graph and sample images to TensorBoard...")
    # Get a single batch from the dataloader to log graph and images
    images, labels = next(iter(training_dataloader))
    grid = torchvision.utils.make_grid(images)

    writer.add_image('Sample Training Images', grid)
    # Ensure the model and input tensor are on the same device for add_graph
    writer.add_graph(model, images.to(device))
    print("Done.")

    # ---------------------------------------------------
    # 6. MAIN TRAINING LOOP
    # ---------------------------------------------------
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()

        # Train for one epoch
        avg_loss, accuracy = training(model, training_dataloader, loss_function, optimizer, device) #tells you if optimizer is doing its job on the data it sees
        val_loss, val_acc = evaluate(model, val_loader) #tells you if the network is generalizing; early stopping

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

        # Log weight histograms
        writer.add_histogram('fc.weight', model.classifier.weight, epoch)

    print("\nTraining Finished!")

        
    # ---------------------------------------------------
    # 7. CLEANUP
    # ---------------------------------------------------

    # Close the TensorBoard writer
    writer.close()

    # To view the logs, open a terminal in your project's root directory and run:
    # tensorboard --logdir=runs

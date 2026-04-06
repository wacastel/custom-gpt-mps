import torch
import torch.optim as optim
import math
import os
import glob
from tqdm import tqdm
from model import LanguageModel
from data import dataloader, VOCAB_SIZE, CONTEXT_LENGTH

# --- V2 Hyperparameters (124M Parameter Class) ---
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 12
LEARNING_RATE = 4e-4
MAX_STEPS = 5000
EVAL_INTERVAL = 100
CHECKPOINT_INTERVAL = 1000

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Starting V2 Training on device: {device}")

    # 1. Initialize the Model
    print("Initializing 124M Parameter Language Model...")
    model = LanguageModel(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        n_heads=N_HEADS, 
        n_layers=N_LAYERS, 
        max_seq_len=CONTEXT_LENGTH
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # 2. Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Ensure a directory exists for our checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    # 3. Checkpoint Resumption Logic
    start_step = 0
    checkpoint_files = glob.glob("checkpoints/v2_model_step_*.pt")
    
    if checkpoint_files:
        # Find the checkpoint with the highest step number
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"🔄 Found existing checkpoint. Resuming from: {latest_checkpoint}")
        
        # Load the weights
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
        
        # Extract the step number to resume the progress bar correctly
        start_step = int(latest_checkpoint.split('_')[-1].split('.')[0])
    else:
        print("✨ No checkpoints found. Starting fresh.")

    # 4. The Continuous Streaming Training Loop
    model.train()
    print("\n--- Beginning V2 Pre-Training ---")
    
    # Setup the tqdm progress bar
    pbar = tqdm(total=MAX_STEPS, initial=start_step, desc="Training")
    
    for step_offset, (inputs, targets) in enumerate(dataloader):
        current_step = start_step + step_offset
        
        if current_step >= MAX_STEPS:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits, loss = model(inputs, targets=targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update the progress bar by 1 step
        pbar.update(1)

        # Logging metrics to the progress bar
        if current_step % EVAL_INTERVAL == 0:
            perplexity = math.exp(loss.item())
            # set_postfix displays these metrics neatly at the end of the loading bar
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Perplexity': f"{perplexity:.2f}"})

        # Checkpointing
        MAX_CHECKPOINTS = 3  # Keep the 3 most recent checkpoints
        
        if current_step > 0 and current_step % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"checkpoints/v2_model_step_{current_step}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Rolling Checkpoint Cleanup
            checkpoints = glob.glob("checkpoints/v2_model_step_*.pt")
            # Sort files by the step number extracted from the filename
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # If we have more than our limit, delete the oldest ones
            while len(checkpoints) > MAX_CHECKPOINTS:
                oldest_checkpoint = checkpoints.pop(0)
                try:
                    os.remove(oldest_checkpoint)
                    tqdm.write(f"🧹 Rolling cleanup: Removed {oldest_checkpoint}")
                except OSError as e:
                    tqdm.write(f"Error deleting {oldest_checkpoint}: {e}")

    pbar.close()

    # 5. Final Save & Cleanup
    final_path = "checkpoints/v2_model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Training complete! Final model saved to {final_path}")

    # Cleanup interim checkpoints
    print("🧹 Cleaning up interim checkpoints...")
    cleanup_files = glob.glob("checkpoints/v2_model_step_*.pt")
    for f in cleanup_files:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error deleting {f}: {e}")
    print("✨ Cleanup complete.")

if __name__ == "__main__":
    train()

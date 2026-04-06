import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import tiktoken

# --- V2 Configuration ---
CONTEXT_LENGTH = 1024  # Doubled from V1 to allow for longer reasoning
BATCH_SIZE = 8         # Keeping this conservative; we can scale it up later based on your 48GB RAM
VOCAB_SIZE = 50257     # Standard GPT-2 vocabulary size

class StreamingTextDataset(IterableDataset):
    def __init__(self, dataset_name, split, context_length):
        # streaming=True is the magic word. It downloads the dataset chunk by chunk
        # as the neural network requests it, preventing RAM explosion.
        print(f"Connecting to {dataset_name} stream...")
        self.dataset = load_dataset(dataset_name, name="sample-10BT", split=split, streaming=True)
        self.context_length = context_length
        
        # Initialize the production-grade OpenAI tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token = self.tokenizer.eot_token

    def __iter__(self):
        buffer = []
        for item in self.dataset:
            # Grab the raw text from the web document
            text = item["text"]
            
            # Encode the text into integers and append the End-Of-Text token
            tokens = self.tokenizer.encode(text, allowed_special="all")
            buffer.extend(tokens)
            buffer.append(self.eot_token)

            # Once our buffer has enough tokens for a full context window + 1 target token
            while len(buffer) >= self.context_length + 1:
                # Slice out the perfect sequence length
                chunk = buffer[:self.context_length + 1]
                
                # Pop those tokens off the front of the buffer so we don't reuse them
                buffer = buffer[self.context_length:] 
                
                # Create our shifted inputs (x) and targets (y)
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                
                yield x, y

# Initialize the streaming dataset using the 10 Billion Token sample of FineWeb-Edu
train_dataset = StreamingTextDataset(
    dataset_name="HuggingFaceFW/fineweb-edu", 
    split="train", 
    context_length=CONTEXT_LENGTH
)

# DataLoader setup
# Note: Because we are using an IterableDataset, shuffle=True is handled 
# differently, so we omit it here and stream linearly.
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

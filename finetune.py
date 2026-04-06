import torch
import torch.optim as optim
import math
import random
from datasets import load_dataset
from tokenizers import Tokenizer
from model import LanguageModel
from data import VOCAB_SIZE, CONTEXT_LENGTH

# --- Fine-Tuning Configuration ---
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
BATCH_SIZE = 16
# Notice the Learning Rate is 10x smaller than pre-training!
LEARNING_RATE = 5e-5 
EPOCHS = 1
EVAL_INTERVAL = 100

PRETRAINED_MODEL_PATH = "tinystories_model_v1.pt"
FINETUNED_MODEL_PATH = "tinystories_chat_v1.pt"
TOKENIZER_PATH = "tinystories-bpe.json"

def create_chat_prompt(story_text):
    # Isolate just the first sentence to avoid cross-sentence garbage
    first_sentence = story_text.split('.')[0]
    words = first_sentence.split()
    
    # Filter out stories that are too short to extract a meaningful subject
    if len(words) < 5:
        return None
    
    # Smarter extraction: skip the introductory words (e.g., "Once upon a time there was a")
    # and grab the end of the first sentence, which usually contains the main subject.
    topic = " ".join(words[-4:]).strip(",.")
    
    # Introduce prompt diversity so the model learns intent, not just string matching
    templates = [
        f"Tell me a story about {topic}.",
        f"I want to hear a tale involving {topic}.",
        f"Can you write a short bedtime story about {topic}?",
        f"Make up a story where the main subject is {topic}.",
        f"Write a story focusing on {topic}."
    ]
    
    prompt = random.choice(templates)
    return f"<|user|>\n{prompt}\n<|assistant|>\n{story_text}"

def finetune():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Starting Phase 2: Instruction Tuning on {device}")

    # 1. Load Tokenizer and Pre-trained Model
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    model = LanguageModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, CONTEXT_LENGTH).to(device)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    
    # 2. Setup Dataset for Fine-Tuning
    print("Preparing chat-formatted dataset...")
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:2%]") # We only need a small subset for alignment
    
    chat_tokens = []
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    
    for item in raw_dataset:
        chat_text = create_chat_prompt(item["text"])
        if chat_text:
            encoded = tokenizer.encode(chat_text).ids + [eot_id]
            chat_tokens.extend(encoded)
            
    chat_tokens = torch.tensor(chat_tokens, dtype=torch.long)
    print(f"Total conversational tokens: {len(chat_tokens):,}")

    # 3. Training Loop Setup
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    model.train()

    print("\n--- Beginning Alignment ---")
    step = 0
    # Process in chunks of CONTEXT_LENGTH
    for i in range(0, len(chat_tokens) - (BATCH_SIZE * CONTEXT_LENGTH), BATCH_SIZE * CONTEXT_LENGTH):
        # Create batches
        chunk = chat_tokens[i : i + (BATCH_SIZE * CONTEXT_LENGTH) + 1]
        
        # Ensure the chunk is exactly the right size to be reshaped
        if len(chunk) < (BATCH_SIZE * CONTEXT_LENGTH) + 1:
            break
            
        x = chunk[:-1].view(BATCH_SIZE, CONTEXT_LENGTH).to(device)
        y = chunk[1:].view(BATCH_SIZE, CONTEXT_LENGTH).to(device)

        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            perplexity = math.exp(loss.item())
            print(f"Step {step:04d} | Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f}")
        step += 1

    print("\nInstruction Tuning complete! Saving chat model...")
    torch.save(model.state_dict(), FINETUNED_MODEL_PATH)
    print(f"✅ Chat model saved as '{FINETUNED_MODEL_PATH}'")

if __name__ == "__main__":
    finetune()

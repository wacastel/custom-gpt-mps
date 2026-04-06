import torch
import torch.nn.functional as F
import tiktoken
from model import LanguageModel
from data import VOCAB_SIZE, CONTEXT_LENGTH

# --- V2 Configuration ---
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 12
MODEL_PATH = "checkpoints/v2_model_final.pt"

def load_model(device):
    print("Loading OpenAI tiktoken...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("Loading 124M model architecture...")
    model = LanguageModel(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        n_heads=N_HEADS, 
        n_layers=N_LAYERS, 
        max_seq_len=CONTEXT_LENGTH
    ).to(device)
    
    print(f"Loading trained weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() 
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=40, device="cpu"):
    # Encode prompt using tiktoken
    input_ids = torch.tensor(tokenizer.encode(prompt, allowed_special="all"), dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"\n--- Generating ---")
    print(prompt, end="", flush=True)

    with torch.no_grad(): 
        for _ in range(max_new_tokens):
            context = input_ids[:, -CONTEXT_LENGTH:]
            logits, _ = model(context)
            
            next_token_logits = logits[0, -1, :]
            next_token_logits = next_token_logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eot_token:
                break
                
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            
            # tiktoken decode requires a list
            try:
                word = tokenizer.decode([next_token.item()])
                print(word, end="", flush=True)
            except UnicodeDecodeError:
                # Handle sub-byte token fragments safely
                continue
            
    print("\n------------------\n")

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        model, tokenizer = load_model(device)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {MODEL_PATH}.")
        exit(1)
        
    print("✅ Model loaded successfully!")
    
    while True:
        user_prompt = input("Enter a document prefix to complete (or 'quit' to exit): ")
        if user_prompt.lower() in ['quit', 'exit']:
            break
            
        generate(model, tokenizer, user_prompt, device=device)
    
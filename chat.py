import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import LanguageModel
from data import VOCAB_SIZE, CONTEXT_LENGTH

# --- Configuration ---
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
MODEL_PATH = "tinystories_chat_v1.pt" # Loading the fine-tuned weights!
TOKENIZER_PATH = "tinystories-bpe.json"

def load_chat_model(device):
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    print("Loading fine-tuned chat model...")
    model = LanguageModel(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        n_heads=N_HEADS, 
        n_layers=N_LAYERS, 
        max_seq_len=CONTEXT_LENGTH
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() 
    return model, tokenizer

def chat(model, tokenizer, device):
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    
    print("\n" + "="*50)
    print("🤖 TinyStories Assistant is online.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Shutting down...")
            break
            
        # 1. Format the user's input using our synthetic template
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        
        # 2. Encode and prepare for generation
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)
        
        print("Assistant: ", end="", flush=True)

        # 3. Autoregressive generation
        with torch.no_grad():
            for _ in range(200): # Allow up to 200 tokens for the response
                context = input_ids[:, -CONTEXT_LENGTH:]
                logits, _ = model(context)
                
                next_token_logits = logits[0, -1, :]
                
                # Temperature scaling (slightly lower for chat to keep it grounded)
                temperature = 0.7 
                next_token_logits = next_token_logits / temperature
                
                # Top-K filtering
                top_k = 40
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 4. Stop if the model decides its thought is complete
                if next_token.item() == eot_id:
                    break
                    
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                
                word = tokenizer.decode([next_token.item()])
                print(word, end="", flush=True)
                
        print("\n" + "-"*50)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        model, tokenizer = load_chat_model(device)
        chat(model, tokenizer, device)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {MODEL_PATH}.")
        exit(1)

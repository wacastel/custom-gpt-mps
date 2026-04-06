import torch
import torch.nn.functional as F
import tiktoken
from model import LanguageModel
from data import VOCAB_SIZE, CONTEXT_LENGTH

# --- V2 Configuration (124M Parameters) ---
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 12
MODEL_PATH = "checkpoints/v2_model_final.pt" # Update this to your fine-tuned weights later

def load_chat_model(device):
    print("Loading OpenAI tiktoken...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("Loading 124M chat model...")
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
    print("\n" + "="*50)
    print("🤖 Custom GPT Assistant is online.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Shutting down...")
            break
            
        # Format the user's input using our synthetic template
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        
        # Encode and prepare for generation
        input_ids = torch.tensor(tokenizer.encode(prompt, allowed_special="all"), dtype=torch.long).unsqueeze(0).to(device)
        
        print("Assistant: ", end="", flush=True)

        with torch.no_grad():
            for _ in range(300): # Allow up to 300 tokens for longer responses
                context = input_ids[:, -CONTEXT_LENGTH:]
                logits, _ = model(context)
                
                next_token_logits = logits[0, -1, :]
                
                temperature = 0.7 
                next_token_logits = next_token_logits / temperature
                
                top_k = 40
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == tokenizer.eot_token:
                    break
                    
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                
                try:
                    word = tokenizer.decode([next_token.item()])
                    print(word, end="", flush=True)
                except UnicodeDecodeError:
                    continue
                
        print("\n" + "-"*50)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        model, tokenizer = load_chat_model(device)
        chat(model, tokenizer, device)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {MODEL_PATH}.")
        exit(1)

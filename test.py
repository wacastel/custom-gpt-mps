import tiktoken
from data import dataloader

def test_v2_pipeline():
    print("\n--- Testing V2 Streaming Data Pipeline ---")
    print("Fetching the first batch from HuggingFace (this might take a few seconds)...")
    
    # 1. Grab one batch from the streaming dataloader
    batch_iter = iter(dataloader)
    inputs, targets = next(batch_iter)
    
    # 2. Verify shapes
    print(f"\nInput tensor shape: {inputs.shape}")   # Expected: [8, 1024]
    print(f"Target tensor shape: {targets.shape}") # Expected: [8, 1024]
    
    # 3. Verify shifting with the OpenAI tokenizer
    print("\nDecoding the first 10 tokens of the first sequence:")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    sample_input = inputs[0]
    sample_target = targets[0]
    
    for i in range(10):
        in_token = sample_input[i].item()
        out_token = sample_target[i].item()
        
        # Tiktoken decode takes a list of tokens. 
        # We replace newlines with a string representation so it prints cleanly on one line.
        try:
            in_text = tokenizer.decode([in_token]).replace('\n', '\\n')
            out_text = tokenizer.decode([out_token]).replace('\n', '\\n')
        except UnicodeDecodeError:
            # Tiktoken sometimes splits emojis or complex characters across multiple tokens.
            in_text = "<partial_byte>"
            out_text = "<partial_byte>"
            
        print(f"Step {i}: Model sees '{in_text}' (ID: {in_token}) --> Must predict '{out_text}' (ID: {out_token})")
        
    print("\n✅ V2 Data Pipeline Validation Complete!")

if __name__ == "__main__":
    test_v2_pipeline()

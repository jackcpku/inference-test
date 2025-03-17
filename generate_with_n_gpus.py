import sys
import json
import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import hashlib

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Parse command-line arguments: number of GPUs, run index, and Ethereum address
n = int(sys.argv[1])  # Number of GPUs
run_idx = int(sys.argv[2])  # Run index for multiple runs
address = sys.argv[3]  # Ethereum address as seed (e.g., "0x95222290dd7278aa3ddd389cc1e1d165cc4bafe5")

model_name = "Qwen/QwQ-32B"
prompt = "How many r's are in the word \"strawberry\"?"

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
)
num_gpus = torch.cuda.device_count()
total_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)}
max_memory = {i: total_memory[i] if i < n else 0 for i in range(num_gpus)}

# Manually specify device map to balance memory usage
device_map = {i: f"{i*100//n}-{(i+1)*100//n}%" for i in range(n)}

# Function to convert Ethereum address to integer seed
def address_to_seed(address):
    # Clean the address: remove "0x" and convert to lowercase for consistency
    if address.startswith("0x"):
        address = address[2:].lower()
    else:
        address = address.lower()
    
    try:
        # Compute SHA-256 hash of the address string
        hash_obj = hashlib.sha256(address.encode('utf-8'))
        hash_bytes = hash_obj.digest()  # Get 32 bytes (256 bits)
        # Take the first 8 bytes (64 bits) and convert to an integer
        seed = int.from_bytes(hash_bytes[:8], 'big')  # 'big' for big-endian
        return seed
    except Exception as e:
        raise ValueError(f"Invalid Ethereum address: {e}")

try:
    print(f"Run {run_idx}: Clearing GPU memory before loading model with {n} GPU(s)...")
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Run {run_idx}: Loading model with {n} GPU(s)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        max_memory=max_memory
    )
    print(f"Run {run_idx}: Model loaded successfully with {n} GPU(s).")

    # Set seed for reproducibility using the Ethereum address
    seed = address_to_seed(address)
    torch.manual_seed(seed)
    temperature = 1.5  # Temperature for softmax sampling

    # Prepare input tensors
    model_inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    generated_ids = input_ids.clone()

    max_new_tokens = 100
    eos_token_id = tokenizer.eos_token_id

    # Manual generation loop
    for step in range(max_new_tokens):
        # Forward pass to get logits
        outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Logits for the next token (1, vocab_size)
        
        # Probabilistic sampling: convert logits to probabilities and sample
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        selected_token_id = next_token.item()
        selected_logit = logits[0, selected_token_id].item()
        selected_prob = probs[0, selected_token_id].item()

        # Append the selected token
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device)],
            dim=1
        )

        # Print top-5 logits and tokens with probabilities
        top_k = 5
        top_logits, top_indices = torch.topk(logits, k=top_k, dim=-1)
        top_probs = probs[0, top_indices[0]].tolist()
        print(f"Run {run_idx}, Step {step} (with {n} GPUs): Top {top_k} tokens:")
        selected_in_top_k = False
        for i in range(top_k):
            token_id = top_indices[0, i].item()
            token = tokenizer.decode([token_id])
            logit = top_logits[0, i].item()
            prob = top_probs[i]
            if token_id == selected_token_id:
                print(f"  {token} (id: {token_id}, logit: {logit:.4f}, prob: {prob:.4f}) <- selected")
                selected_in_top_k = True
            else:
                print(f"  {token} (id: {token_id}, logit: {logit:.4f}, prob: {prob:.4f})")
        if not selected_in_top_k:
            selected_token = tokenizer.decode([selected_token_id])
            print(f"  Selected: {selected_token} (id: {selected_token_id}, logit: {selected_logit:.4f}, prob: {selected_prob:.4f})")

        # Stop if EOS token is generated
        if selected_token_id == eos_token_id:
            print(f"Run {run_idx}: EOS token generated at step {step} with {n} GPUs")
            break

    # Decode and save the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    result = {"generated_text": generated_text}
    output_file = f"results_{n}_run{run_idx}.json"
    with open(output_file, "w") as f:
        json.dump(result, f)
    print(f"Run {run_idx}: Generated text saved to {output_file}")

except Exception as e:
    output_file = f"results_{n}_run{run_idx}.json"
    with open(output_file, "w") as f:
        json.dump({"error": str(e)}, f)
    print(f"Run {run_idx}: Error with {n} GPUs: {str(e)}")

finally:
    # Clean up to free memory
    if "model" in locals():
        del model
    if "model_inputs" in locals():
        del model_inputs
    if "generated_ids" in locals():
        del generated_ids
    gc.collect()
    torch.cuda.empty_cache()
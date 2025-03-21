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

# Load tokenizer and compute token mapping
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokens = [tokenizer.convert_ids_to_tokens(i) for i in range(tokenizer.vocab_size)]

# Prepare input text using chat template
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
)

# Configure device mapping based on number of GPUs
num_gpus = torch.cuda.device_count()
total_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)}
max_memory = {i: total_memory[i] if i < n else 0 for i in range(num_gpus)}
device_map = {i: f"{i*100//n}-{(i+1)*100//n}%" for i in range(n)}

# Function to convert Ethereum address to integer seed
def address_to_seed(address):
    if address.startswith("0x"):
        address = address[2:].lower()
    else:
        address = address.lower()
    try:
        hash_obj = hashlib.sha256(address.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        seed = int.from_bytes(hash_bytes[:8], 'big')
        return seed
    except Exception as e:
        raise ValueError(f"Invalid Ethereum address: {e}")

try:
    # Clear memory before loading model
    if n > 0:
        print(f"Run {run_idx}: Clearing GPU memory before loading model with {n} GPU(s)...")
        torch.cuda.empty_cache()
    else:
        print(f"Run {run_idx}: Preparing to load model on CPU...")
    gc.collect()

    # Load model
    if n > 0:
        print(f"Run {run_idx}: Loading model with {n} GPU(s)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            max_memory=max_memory
        )
    else:
        print(f"Run {run_idx}: Loading model on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu"
        )
    print(f"Run {run_idx}: Model loaded successfully{' with ' + str(n) + ' GPU(s)' if n > 0 else ' on CPU'}.")

    # Set seed for reproducibility
    seed = address_to_seed(address)
    torch.manual_seed(seed)
    temperature = 0.1

    # Prepare input tensors
    model_inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    generated_ids = input_ids.clone()

    max_new_tokens = 100
    eos_token_id = tokenizer.eos_token_id

    # Initialize list to store generation data
    steps = []

    # Manual generation loop
    for step in range(max_new_tokens):
        # Compute logits
        outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)

        # Store logits for this step
        steps.append({
            "logits": logits[0].tolist(),  # Convert to list
            "selected_token_id": None  # Placeholder
        })

        # Sample next token
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: (1, 1)
        selected_token_id = next_token.item()
        steps[-1]["selected_token_id"] = selected_token_id  # Update selected token ID

        selected_logit = logits[0, selected_token_id].item()
        selected_prob = probs[0, selected_token_id].item()

        # Update generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device)],
            dim=1
        )

        # Print top-k tokens
        top_k = 20
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

        # Check for EOS token
        if selected_token_id == eos_token_id:
            print(f"Run {run_idx}: EOS token generated at step {step} with {n} GPUs")
            break

    # Save generation data to JSON
    logits_file = f"logits_{n}_run{run_idx}.json"
    data = {
        "vocab_size": tokenizer.vocab_size,
        "tokens": tokens,
        "steps": steps
    }
    with open(logits_file, "w") as f:
        json.dump(data, f)
    print(f"Run {run_idx}: Logits data saved to {logits_file}")

    # Save generated text
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
    # Clean up memory
    if "model" in locals():
        del model
    if "model_inputs" in locals():
        del model_inputs
    if "generated_ids" in locals():
        del generated_ids
    gc.collect()
    torch.cuda.empty_cache()
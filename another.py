# Import necessary libraries
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configuration: Manually set seed and temperature
seed = 3407  # Seed for reproducibility
temperature = 1.5  # Temperature for controlling randomness (lower = less random)
prompt = "How many r's are in the word \"strawberry\"?"  # Test prompt

# Set the seed for reproducibility across CPU and all GPUs
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Define the model name
model_name = "Qwen/QwQ-32B"

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded.")

torch.cuda.empty_cache()
gc.collect()

# Load the model, distributing it across available GPUs
print("Loading model...")
num_gpus = torch.cuda.device_count()
total_memory = {i: torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)}
max_memory = {i: total_memory[i] for i in range(num_gpus)}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    max_memory=max_memory
)
print("Model loaded.")

# Prepare the input using the chat template
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)

# Define generation parameters
max_new_tokens = 64  # Maximum number of new tokens to generate
top_k = 40  # Top-k sampling
top_p = 0.9999  # Top-p (nucleus) sampling
repetition_penalty = 1.1  # Penalty to reduce repetition

# Generate the response with detailed output
print("Generating response...")
generated_output = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,  # Include attention mask for proper generation
    max_new_tokens=max_new_tokens,
    do_sample=True,  # Enable sampling for temperature to take effect
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    repetition_penalty=repetition_penalty,
    return_dict_in_generate=True,  # Return a GenerateOutput object
    output_scores=True  # Include logits for each step
)
print("Generation complete.")

# Extract generated token IDs
input_length = model_inputs.input_ids.shape[1]
generated_ids = generated_output.sequences[0, input_length:]

# Print step-by-step information
for i in range(len(generated_ids)):
    # Get the logits for this step (shape: [vocab_size])
    logits = generated_output.scores[i][0]
    
    # Apply temperature scaling (since scores are logits post-processors but pre-warpers)
    logits_scaled = logits / temperature
    
    # Compute probabilities using softmax
    probs = torch.softmax(logits_scaled, dim=-1)
    
    # Get the top 5 probabilities and their corresponding token IDs
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    # Get the generated token ID for this step
    gen_token_id = generated_ids[i].item()
    
    # Decode the generated token
    gen_token = tokenizer.decode([gen_token_id], skip_special_tokens=False)
    
    # Print the step information
    print(f"\nStep {i+1}:")
    print(f"Generated token: '{gen_token}' (id: {gen_token_id})")
    print("Top 5 candidates:")
    
    # Print the top 5 candidates with logits and probabilities
    for j in range(5):
        token_id = top5_indices[j].item()
        token_prob = top5_probs[j].item()
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        token_logit = logits[token_id].item()  # Original logit value
        print(f"  {j+1}. '{token_str}' (id: {token_id}): logit={token_logit:.4f}, prob={token_prob:.4f}")

# Decode and print the entire output
output = tokenizer.decode(generated_output.sequences[0], skip_special_tokens=True)
print("\nEntire output:")
print(output)
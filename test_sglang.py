import torch
import sglang as sgl
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def main():
    # Configuration
    seed = 3408
    temperature = 0
    prompt = "How many r's are in the word \"strawberry\"?"
    model_name = "Qwen/QwQ-32B"
    tp_size = 4
    max_new_tokens = 100
    top_k = 40
    top_p = 0.95
    repetition_penalty = 1.1

    # Set environment variable
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load tokenizer and compute token mapping
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in range(tokenizer.vocab_size)]
    print("Tokenizer loaded.")

    # Load model for output layer
    print("Loading model for output layer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    output_weight = model.lm_head.weight.to("cuda:0")  # Move to GPU
    del model  # Free CPU memory
    print("Output layer loaded.")

    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Create sglang engine
    print("Loading model with sglang...")
    llm = sgl.Engine(model_path=model_name, tp_size=tp_size)
    print("Model loaded.")

    # Define sampling parameters
    sampling_params = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
    }

    # Generate response with hidden states
    print("Generating response...")
    outputs = llm.generate([text], sampling_params=sampling_params, return_hidden_states=True)
    print("Generation complete.")

    # Process output
    output = outputs[0]
    generated_text = output["text"]
    hidden_states = output["meta_info"]["hidden_states"]

    # Convert hidden states to torch tensors on GPU
    hidden_states = [
        torch.tensor(hs, dtype=torch.bfloat16, device="cuda:0") for hs in hidden_states
    ]

    # Get generated token IDs
    generated_ids = output["meta_info"].get(
        "output_ids", tokenizer.encode(generated_text, add_special_tokens=False)
    )

    # Verify lengths
    print(f"Number of hidden states: {len(hidden_states)}")
    print(f"Number of generated tokens: {len(generated_ids)}")

    # Initialize list to store generation data
    steps = []

    # Process each step
    for step, hidden_state in enumerate(hidden_states):
        print(f"Step {step}: Shape of hidden_state: {hidden_state.shape}")
        if hidden_state.dim() == 1:
            hs = hidden_state  # Shape: [hidden_size]
        elif hidden_state.dim() == 2:
            hs = hidden_state[-1, :]  # Shape: [hidden_size]
        else:
            raise ValueError(f"Unexpected hidden_state shape: {hidden_state.shape}")

        # Compute logits
        logits = hs @ output_weight.T  # Shape: [vocab_size]

        # Store logits and selected token ID
        steps.append({
            "logits": logits.tolist(),  # Convert to list
            "selected_token_id": generated_ids[step]
        })

        # Print top 20 tokens (optional)
        top20_values, top20_indices = torch.topk(logits, 20)
        top20_tokens = [tokenizer.decode([idx]) for idx in top20_indices.tolist()]
        actual_token_id = generated_ids[step]
        actual_token = tokenizer.decode([actual_token_id])
        print(f"\nStep {step}:")
        print(f"  Actual token: '{actual_token}'")
        print(f"  Top 20 candidates: {top20_tokens}")
        print(f"  Top 20 logits: {top20_values.tolist()}")

    # Save generation data to JSON
    data = {
        "vocab_size": tokenizer.vocab_size,
        "tokens": tokens,
        "steps": steps
    }
    with open("logits_sglang_run10100.json", "w") as f:
        json.dump(data, f)
    print("Logits data saved to logits_sglang_run10100.json")

    # Shutdown engine
    llm.shutdown()

if __name__ == "__main__":
    main()
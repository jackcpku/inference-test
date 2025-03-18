import torch
import sglang as sgl
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Configuration
    seed = 3408
    temperature = 0
    prompt = "How many r's are in the word \"strawberry\"?"
    model_name = "Qwen/QwQ-32B"
    tp_size = 4
    max_new_tokens = 64
    top_k = 40
    top_p = 0.95
    repetition_penalty = 1.1

    # Set environment variable
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")

    # Load model to get output layer
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

    # Generate response
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

    # Verify hidden states length matches generated tokens
    print(f"Number of hidden states: {len(hidden_states)}")
    print(f"Number of generated tokens: {len(generated_ids)}")

    # Print top 5 logits for each step
    for step, hidden_state in enumerate(hidden_states):
        # Handle different hidden state shapes
        if hidden_state.dim() == 1:
            hs = hidden_state  # Shape: [hidden_size]
        elif hidden_state.dim() == 2:
            hs = hidden_state[-1, :]  # Take last layer, Shape: [hidden_size]
        else:
            raise ValueError(f"Unexpected hidden_state shape: {hidden_state.shape}")
        
        # Compute logits
        logits = hs @ output_weight.T  # [hidden_size] @ [hidden_size, vocab_size] -> [vocab_size]
        
        # Get top 5 predictions
        top5_values, top5_indices = torch.topk(logits, 20)
        top5_tokens = [tokenizer.decode([idx]) for idx in top5_indices.tolist()]
        
        # Actual generated token
        actual_token_id = generated_ids[step]
        actual_token = tokenizer.decode([actual_token_id])
        
        # Print results
        print(f"\nStep {step}:")
        print(f"  Actual token: '{actual_token}'")
        print(f"  Top 5 candidates: {top5_tokens}")
        print(f"  Top 5 logits: {top5_values.tolist()}")
        # Print full output
        print("\nEntire output:")
        print(generated_text)

    # Shutdown
    llm.shutdown()

if __name__ == "__main__":
    main()
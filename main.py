from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import time
import torch  # Added to access GPU information

# Define the model name
model_name = "Qwen/QwQ-32B"

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully.")

# Load the model with automatic device mapping
print("Loading model (this may take some time due to model size)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Automatically select precision (e.g., float16)
    device_map="auto"    # Distribute model across all GPUs
)
print("Model loaded successfully.")

# Check the number of GPUs available to PyTorch
print(f"Number of GPUs available: {torch.cuda.device_count()}")

# Check how the model is distributed across GPUs
print("Model device map:", model.hf_device_map)

# Define the prompt
prompt = "How many r's are in the word \"strawberry\"?"
messages = [{"role": "user", "content": prompt}]

# Prepare the input using the chat template
print("Preparing input...")
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)

# Custom streamer to print one character at a time
class CharByCharStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):
        for char in text:
            print(char, end='', flush=True)
            time.sleep(0.05)  # Optional delay to simulate typing
        if stream_end:
            print()  # New line at the end

# Generate a response with the custom streamer
print("Generating response...")
streamer = CharByCharStreamer(tokenizer)
generated_ids = model.generate(**model_inputs, max_new_tokens=512, streamer=streamer)
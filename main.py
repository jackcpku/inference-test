from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, LogitsProcessor, LogitsProcessorList
import time
import torch

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

# Custom logits processor to print top k tokens
class TopKPrinter(LogitsProcessor):
    def __init__(self, tokenizer, k=10):
        self.tokenizer = tokenizer
        self.k = k
        self.step = 0

    def __call__(self, input_ids, scores):
        self.step += 1
        print(f"\nStep {self.step}:")
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(scores, dim=-1)
        # Get top k tokens and their probabilities
        top_k_probs, top_k_indices = torch.topk(probabilities, self.k, dim=-1)
        top_k_tokens = [self.tokenizer.decode(idx.item()) for idx in top_k_indices[0]]
        # Print in a structured format
        print(f"{'Token':<15} Probability")
        for token, prob in zip(top_k_tokens, top_k_probs[0]):
            print(f"{token:<15} : {prob.item():.4f}")
        print("-" * 20)
        return scores  # Return logits unchanged

# Create the logits processor
printer = TopKPrinter(tokenizer, k=10)
logits_processor = LogitsProcessorList([printer])

# Custom streamer to print one character at a time
class CharByCharStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):
        for char in text:
            print(char, end='', flush=True)
            time.sleep(0.05)  # Optional delay to simulate typing
        if stream_end:
            print()  # New line at the end

# Generate a response with the custom streamer and logits processor
print("Generating response...")
streamer = CharByCharStreamer(tokenizer)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
    logits_processor=logits_processor,
    streamer=streamer
)
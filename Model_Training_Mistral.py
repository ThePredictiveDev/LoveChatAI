
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, pipeline
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import json

torch.backends.cuda.enable_flash_sdp(True)

TOKEN = '' #Your huggingface_hub token
login(TOKEN)

# Define model name
model_name = "mistralai/Mistral-7B-v0.1"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set quantization configuration for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model in 4-bit mode
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)


# Load the dataset
dataset_path = "structured_chat_data.json"  # Update this path if needed
dataset = load_dataset("json", data_files=dataset_path, field="messages", split="train")

# Print a sample to verify
print(dataset[0])

tokenizer.pad_token = tokenizer.eos_token
# Tokenizer function
def tokenize_chat(example):
    if "role" not in example or "content" not in example:
        print(f"❌ Skipping invalid entry: {example}")
        return {"input_ids": []}

    text = f"{example['role'].capitalize()}: {example['content']}\n"

    return tokenizer(text, truncation=True, padding="max_length", max_length=512)



# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_chat)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank for LoRA
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # LoRA on attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
device = 'cuda'
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(device)
# Define training parameters
training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="paged_adamw_32bit",
    save_total_limit=2,
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    report_to="none",
    save_steps=1000
)

# Initialize Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Handle extra arguments
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss
    
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Start fine-tuning
trainer.train()

model.save_pretrained("./mistral-7b-finetuned")
tokenizer.save_pretrained("./mistral-7b-finetuned")

chatbot = pipeline("text-generation", model="./mistral-7b-finetuned", tokenizer=tokenizer)

# Test on a chat prompt
response = chatbot("User: Heyy\nAI:", max_length=100, do_sample=True, temperature=0.7)
print(response[0]["generated_text"])
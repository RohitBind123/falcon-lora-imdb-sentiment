# falcon-lora-imdb-sentiment
This project demonstrates how to fine-tune the tiiuae/falcon-rw-1b Large Language Model (LLM) using Low-Rank Adaptation (LoRA) on the IMDB sentiment analysis dataset. The entire pipeline is implemented using Hugging Face Transformers, PEFT (Parameter Efficient Fine-Tuning), and is optimized to run in Google Colab.

# 🚀 Falcon-RW-1B Fine-Tuned on IMDB for Sentiment Analysis (LoRA + PEFT)

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Model](https://img.shields.io/badge/model-falcon--rw--1b-yellow)
![HuggingFace](https://img.shields.io/badge/hub-rohitbind%2Ffalcon--lora--imdb-orange)

This project demonstrates how to fine-tune the [`tiiuae/falcon-rw-1b`](https://huggingface.co/tiiuae/falcon-rw-1b) large language model using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) for **sentiment generation** on the IMDB dataset. Training is done in 8-bit using the PEFT + Hugging Face Transformers libraries.

---

## 📌 Project Structure

- Fine-tuning with LoRA on IMDB
- Using `transformers`, `datasets`, `peft`, and `bitsandbytes`
- Uploading to Hugging Face Hub
- Inference with the fine-tuned adapter

---

## 📂 Dataset

- **Name**: `stanfordnlp/imdb`
- **Type**: Movie review dataset
- **Size**: 25,000 labeled movie reviews (train/test)

```python
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb")
```
⚙️ Setup Instructions
```
# Install core libraries
pip uninstall -y torch torchvision torchaudio
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -q --upgrade datasets peft bitsandbytes
```

# Optional: upgrade pip and setuptools
```
pip install -U pip setuptools wheel```
# Login to Hugging Face (for pushing model later)
from huggingface_hub import notebook_login
notebook_login()
```

🧠 Model Loading & Tokenization
```
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
```

🧪 Preprocessing IMDB Dataset
```
def preprocess(data):
    tokens = tokenizer(data["text"], truncation=True, padding='max_length', max_length=128)
    tokens["labels"] = tokens["input_ids"]
    return tokens

train_dataset = dataset["train"].map(preprocess, batched=True)
train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
```
🧬 LoRA Fine-Tuning Setup
```
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    target_modules=["query_key_value"]
)

model = get_peft_model(model, lora_config)
```
🏃 Training with Hugging Face Trainer
```
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./falcon_lora_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_total_limit=1,
    fp16=True,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.select(range(1000))
)

trainer.train()
```
💬 Inference from Fine-Tuned Model

```from transformers import pipeline
from peft import PeftModel, PeftConfig

adapter_path = "./falcon_lora_output/checkpoint-125"
peft_config = PeftConfig.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("The movie was really bad because"))
```

🧾 Example Output
Prompt: The movie was absolutely wonderful because
Generated: it had a fantastic storyline, brilliant performances, and a satisfying ending.

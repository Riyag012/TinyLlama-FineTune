# Fine-Tuned TinyLLaMA: A Specialized AI/ML Chatbot

[](https://huggingface.co/spaces/Riya012/TinyLlama_AI_ML_Chatbot)
[](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
[](https://opensource.org/licenses/MIT)

This project enhances the **TinyLLaMA-1.1B-Chat-v1.0** model by fine-tuning it on a niche,  dataset of 301 questions and answers focused on **Artificial Intelligence, Machine Learning, and Deep Learning**. The result is a specialized chatbot that provides more accurate and context-aware responses within this domain compared to the general-purpose base model.

The model is deployed and accessible via an interactive Gradio web UI on Hugging Face Spaces.

## 🚀 Live Demo

You can interact with the fine-tuned chatbot live on Hugging Face Spaces:

**[➡️ Click here to access the Live Demo](https://huggingface.co/spaces/Riya012/TinyLlama_AI_ML_Chatbot)**

-----

## ✨ Key Features

  * **🧠 Specialized Knowledge**: Expertly fine-tuned on a custom dataset to answer questions about AI, ML, and DL concepts, from foundational topics to advanced models like GANs and Transformers.
  * **⚡ Efficient Fine-Tuning**: Utilizes **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning (PEFT), modifying only a tiny fraction (0.4%) of the model's parameters. This significantly reduces computational costs and training time.
  * **📊 Comprehensive Evaluation**: The model's performance was rigorously evaluated on a dedicated test set using a suite of metrics including Loss, Perplexity, BLEU, ROUGE, and BERTScore to ensure high-quality text generation.
  * **🌐 Interactive & Accessible**: Deployed with a user-friendly Gradio interface, making the powerful model easily accessible to everyone.

-----

## 🛠️ Technical Stack

  * **Base Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  * **Frameworks & Libraries**: PyTorch, Transformers, PEFT (Parameter-Efficient Fine-Tuning), Accelerate, BitsandBytes, TRL (Transformer Reinforcement Learning)
  * **Evaluation**: Evaluate, ROUGE Score, BERT Score, BLEU
  * **Deployment**: Gradio, Hugging Face Spaces & Hub

-----

## ⚙️ Project Workflow

The project followed a systematic workflow from data preparation to deployment:

1.  **Data Preparation**: A custom dataset of 301 Q\&A pairs on AI/ML was created in JSON format. Each entry was structured with `"instruction"` and `"output"` fields. This data was then formatted into the Alpaca instruction-following format (`### Instruction: ... ### Response: ...`).

2.  **Tokenization**: The formatted text dataset was tokenized using the base model's tokenizer. The sequences were padded and truncated to a `max_length` of 512.

3.  **LoRA Configuration**: LoRA was configured to adapt the model efficiently. Adapters were added to the attention mechanism's query, key, value, and output projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`).

4.  **Model Training**: The model was trained for 15 epochs using the `Trainer` API from the Transformers library. Key hyperparameters included a learning rate of `1e-4`, weight decay for regularization, and early stopping to prevent overfitting. The entire training process was logged to Weights & Biases (W\&B) for monitoring.

5.  **Model Evaluation**: After training, the model was evaluated on an unseen test split of the dataset. The final metrics demonstrate the model's strong performance in understanding and generating domain-specific text.

6.  **Deployment**: The final LoRA adapter weights were pushed to the Hugging Face Hub. A Gradio application was built to provide an interactive chat interface and deployed on Hugging Face Spaces for public access.

-----

## 📈 Performance & Evaluation

The model was evaluated on the test set after completing the training process. The final results are as follows:

| Metric                | Score      | Description                                                              |
| --------------------- | ---------- | ------------------------------------------------------------------------ |
| **`eval_loss`** | `0.7143`   | The final cross-entropy loss on the test set. Lower is better.           |
| **`eval_perplexity`** | `5.8517`   | A measure of how well the model predicts the text. Lower is better.      |
| **`eval_token_accuracy`** | `0.6012`   | The accuracy of predicting the next token. Higher is better.             |
| **`eval_bleu`** | `0.2398`   | Measures n-gram overlap with reference text (precision-focused).         |
| **`eval_rouge1`** | `0.5498`   | Measures unigram overlap with reference text (recall-focused).           |
| **`eval_rougeL`** | `0.4593`   | Measures the longest common subsequence between prediction and reference.  |
| **`eval_bertscore_f1`** | `0.8648`   | Measures semantic similarity using BERT embeddings. Higher is better.    |

-----

## 🚀 How to Use the Model

You can easily use this fine-tuned model for inference directly from the Hugging Face Hub.

### 1\. Install necessary libraries

```bash
pip install torch transformers accelerate bitsandbytes peft
```

### 2\. Run the inference script

The following Python script loads the base TinyLLaMA model and applies the fine-tuned LoRA adapter for inference.

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define model paths
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_model_id = "Riya012/TinyLlama_AI_ML_Chatbot" # Your HF repo

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the LoRA adapter and merge with the base model
model = PeftModel.from_pretrained(base_model, adapter_model_id)
model.eval()

# Prepare the prompt
user_query = "What are Generative Adversarial Networks (GANs)?"
messages = [{"role": "user", "content": user_query}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize and generate a response
inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1
    )

# Decode and print the response
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Clean up the response to show only the assistant's message
assistant_response = response_text.split("<|assistant|>")[1].strip()
print(assistant_response)

```

-----

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

from src.models.hf_loader import load_model_and_tokenizer

models = [
    "bert-base",
    "roberta-base",
    "deberta-v3-base",
    "gte-multilingual-base",
]

for model_name in models:
    print(f"\n===== {model_name} =====")
    tokenizer, model = load_model_and_tokenizer(model_name)
    print("OK")
    print(type(tokenizer))
    print(type(model))
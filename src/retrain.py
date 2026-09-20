import os
import sys
import yaml
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Aggiunge la directory radice al percorso di ricerca dei moduli Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.evaluate import run_evaluation
except ImportError:
    # Fallback se il modulo evaluate è direttamente nella cartella corrente
    from evaluate import run_evaluation

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_retraining():
    print("🚀 [RETRAINING AUTOMATICO] Avvio del processo di Fine-Tuning...")
    config = load_config()
    
    # 1. Caricamento del dataset di retrain / log accumulati
    log_path = "data/predictions_log.csv"
    if not os.path.exists(log_path):
        print("⚠️ Nessun dato di log trovato in 'data/predictions_log.csv'. Generazione batch sintetico per test...")
        os.makedirs("data", exist_ok=True)
        synthetic_data = pd.DataFrame({
            "text": ["Great product!", "Bad experience", "Normal quality"] * 10,
            "label": [2, 0, 1] * 10  # Mapping: 0=Negative, 1=Neutral, 2=Positive
        })
        synthetic_data.to_csv(log_path, index=False)
    
    df = pd.read_csv(log_path)
    # Assicuriamo la presenza delle colonne necessarie
    if "label" not in df.columns and "predicted_label" in df.columns:
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        df["label"] = df["predicted_label"].map(label_map)
    
    dataset = Dataset.from_pandas(df[["text", "label"]])
    
    # 2. Tokenizzazione
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_func, batched=True)
    
    # 3. Caricamento Modello Base
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # 4. Configurazione Addestramento da config.yaml
    training_args = TrainingArguments(
        output_dir="./results_retrain",
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        learning_rate=float(config["training"]["learning_rate"]),
        logging_steps=5,
        save_strategy="no",
        use_cpu=not torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print("🏋️ Inizio Fine-Tuning del modello...")
    trainer.train()
    
    # 5. Salvataggio temporaneo del nuovo checkpoint
    new_model_dir = "./models/retrained_model"
    model.save_pretrained(new_model_dir)
    tokenizer.save_pretrained(new_model_dir)
    print(f"✅ Modello riaddestrato salvato temporaneamente in: {new_model_dir}")
    
    # 6. Valutazione e verifica della soglia di promozione (Quality Gate Retraining)
    print("📊 Valutazione delle performance del nuovo modello...")
    eval_metrics = run_evaluation(config_path="config/config.yaml")
    
    # Recupera la baseline F1 da metrics.json attuale
    current_f1 = 0.6840
    if os.path.exists("data/metrics.json"):
        with open("data/metrics.json", "r") as f:
            old_metrics = json.load(f)
            current_f1 = old_metrics.get("f1_macro", current_f1)
            
    new_f1 = eval_metrics.get("f1_macro", 0.0)
    f1_diff = new_f1 - current_f1
    min_improvement = config["training"].get("min_f1_improvement", 0.005)
    
    print(f"\n=== VERDETTO PROMOZIONE MODELLO ===")
    print(f"• F1-Macro Attuale : {current_f1:.4f}")
    print(f"• F1-Macro Nuovo   : {new_f1:.4f}")
    print(f"• Incremento       : {f1_diff:+.4f} (Richiesto minimo: {min_improvement})")
    
    if f1_diff >= min_improvement:
        print("🎉 PROMOZIONE APPROVATA: Il nuovo modello supera la soglia di miglioramento e viene promosso!")
    else:
        print("⚠️ PROMOZIONE RESPINTA: Il miglioramento è insufficiente. Il modello operativo rimane quello attuale.")

if __name__ == "__main__":
    run_retraining()

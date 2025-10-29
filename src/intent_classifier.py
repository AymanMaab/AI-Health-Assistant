import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentClassifier:
    """Intent classification using DistilBERT"""
    
    def __init__(self):
        self.model_name = settings.INTENT_MODEL_NAME
        self.device = settings.DEVICE
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.intents = settings.INTENT_CATEGORIES
        
    def prepare_dataset(self, df: pd.DataFrame):
        """Prepare dataset for training"""
        from datasets import Dataset
        
        # Encode labels
        df['label'] = self.label_encoder.fit_transform(df['intent'])
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(df[['question', 'label']])
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['question'],
                padding='max_length',
                truncation=True,
                max_length=128
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['question'])
        tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
        tokenized_dataset.set_format('torch')
        
        return tokenized_dataset
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train the intent classifier"""
        logger.info("Training intent classifier...")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.intents)
        )
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df)
        val_dataset = self.prepare_dataset(val_df)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(settings.MODELS_DIR / "intent_classifier"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(settings.MODELS_DIR / "logs"),
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )
        
        # Metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = (predictions == labels).mean()
            return {"accuracy": accuracy}
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Save model
        model_path = settings.MODELS_DIR / "intent_classifier" / "final"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save label encoder
        label_mapping = {i: intent for i, intent in enumerate(self.intents)}
        with open(settings.MODELS_DIR / "intent_classifier" / "labels.json", 'w') as f:
            json.dump(label_mapping, f)
        
        logger.info(f"Intent classifier saved to {model_path}")
        
    def load(self):
        """Load trained model"""
        model_path = settings.MODELS_DIR / "intent_classifier" / "final"
        
        if not model_path.exists():
            logger.warning("No trained model found. Using pretrained model.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.intents)
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Load label mapping
            with open(settings.MODELS_DIR / "intent_classifier" / "labels.json", 'r') as f:
                label_mapping = json.load(f)
                self.intents = [label_mapping[str(i)] for i in range(len(label_mapping))]
        
        self.model.to(self.device)
        self.model.eval()
        logger.info("Intent classifier loaded successfully")
    
    def predict(self, text: str) -> dict:
        """Predict intent for a given text"""
        if self.model is None:
            self.load()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'intent': self.intents[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                intent: float(prob) 
                for intent, prob in zip(self.intents, probabilities[0].cpu().numpy())
            }
        }


if __name__ == "__main__":
    # Test the classifier
    classifier = IntentClassifier()
    
    # Load or create sample data
    train_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "val.csv")
    
    # Train
    classifier.train(train_df, val_df)
    
    # Test prediction
    test_query = "What are the side effects of aspirin?"
    result = classifier.predict(test_query)
    print(f"Query: {test_query}")
    print(f"Predicted Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.2%}")
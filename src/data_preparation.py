import pandas as pd
import json
import requests
from pathlib import Path
from datasets import load_dataset, Dataset
from typing import List, Dict
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparation:
    """Handles data collection and preprocessing for health QA"""
    
    def __init__(self):
        self.raw_dir = settings.RAW_DATA_DIR
        self.processed_dir = settings.PROCESSED_DATA_DIR
        
    def download_medquad_dataset(self) -> pd.DataFrame:
        """Download MedQuAD dataset from HuggingFace"""
        try:
            logger.info("Downloading medical Q&A datasets...")
            
            # Try multiple medical datasets
            datasets_to_try = [
                ("medmcqa", "train[:10000]"),  # Medical entrance exams
                ("pubmed_qa", "train[:5000]"),  # PubMed Q&A
            ]
            
            all_data = []
            
            for dataset_name, split in datasets_to_try:
                try:
                    logger.info(f"Trying {dataset_name}...")
                    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
                    df = pd.DataFrame(dataset)
                    
                    # Standardize column names
                    if 'question' in df.columns and 'answer' in df.columns:
                        pass
                    elif 'question' in df.columns and 'cop' in df.columns:
                        # MedMCQA format
                        df['answer'] = df.apply(lambda row: row.get(f'op{row["cop"]}', row.get('opa', 'Unknown')), axis=1)
                    elif 'QUESTION' in df.columns and 'final_decision' in df.columns:
                        # PubMed QA format
                        df = df.rename(columns={'QUESTION': 'question', 'final_decision': 'answer'})
                    
                    if 'question' in df.columns and 'answer' in df.columns:
                        df['intent'] = df['question'].apply(self._classify_intent)
                        all_data.append(df[['question', 'answer', 'intent']])
                        logger.info(f"✓ Loaded {len(df)} entries from {dataset_name}")
                except Exception as e:
                    logger.warning(f"Could not load {dataset_name}: {e}")
                    continue
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['question'])
                output_path = self.raw_dir / "medical_qa.csv"
                combined_df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(combined_df)} medical Q&A pairs to {output_path}")
                return combined_df
            else:
                raise Exception("No datasets loaded successfully")
                
        except Exception as e:
            logger.error(f"Error downloading datasets: {e}")
            return self._create_sample_data()
    
    def _classify_intent(self, question: str) -> str:
        """Simple keyword-based intent classification for labeling"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['drug', 'medicine', 'medication', 'pill', 'dose']):
            return 'medicine_info'
        elif any(word in question_lower for word in ['nutrition', 'diet', 'food', 'vitamin', 'eat']):
            return 'nutrition_advice'
        elif any(word in question_lower for word in ['symptom', 'diagnose', 'feel', 'pain']):
            return 'symptoms_diagnosis'
        elif any(word in question_lower for word in ['treatment', 'therapy', 'surgery', 'procedure']):
            return 'treatment_procedure'
        else:
            return 'general_faq'
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample training data if downloads fail"""
        logger.info("Creating sample training data...")
        
        sample_data = {
            'question': [
                "What are the side effects of metformin?",
                "How much protein should I eat daily?",
                "What causes diabetes?",
                "What is the treatment for hypertension?",
                "What are symptoms of COVID-19?",
                "Can I take ibuprofen with alcohol?",
                "What foods are high in vitamin D?",
                "How does insulin work?",
                "What is chemotherapy?",
                "What causes headaches?",
                "What is the recommended dose of aspirin?",
                "Are eggs good for health?",
                "What is high blood pressure?",
                "How is pneumonia treated?",
                "What are early signs of heart attack?",
            ],
            'answer': [
                "Common side effects of metformin include nausea, diarrhea, stomach upset, and metallic taste. Serious side effects are rare but may include lactic acidosis.",
                "The recommended daily protein intake is 0.8 grams per kilogram of body weight, or about 46-56 grams for average adults.",
                "Diabetes is caused by insufficient insulin production or insulin resistance, often related to genetics, obesity, and lifestyle factors.",
                "Hypertension treatment includes lifestyle changes (diet, exercise) and medications like ACE inhibitors, beta-blockers, or diuretics.",
                "COVID-19 symptoms include fever, cough, fatigue, loss of taste/smell, shortness of breath, and body aches.",
                "Combining ibuprofen with alcohol can increase risk of stomach bleeding and liver damage. It's best to avoid this combination.",
                "Foods high in vitamin D include fatty fish (salmon, mackerel), egg yolks, fortified milk, and mushrooms exposed to sunlight.",
                "Insulin is a hormone that helps glucose enter cells for energy. It regulates blood sugar levels by signaling cells to absorb glucose.",
                "Chemotherapy uses drugs to kill rapidly dividing cancer cells. It can be given orally, intravenously, or through other routes.",
                "Headaches can be caused by stress, dehydration, lack of sleep, eye strain, sinus issues, or more serious conditions.",
                "The typical adult dose of aspirin for pain relief is 325-650mg every 4-6 hours. For heart protection, 81mg daily is common.",
                "Eggs are nutritious, containing protein, vitamins B12 and D, and healthy fats. Moderate consumption is part of a healthy diet.",
                "High blood pressure (hypertension) is when blood pressure consistently measures 130/80 mmHg or higher.",
                "Pneumonia treatment depends on the cause but typically includes antibiotics for bacterial pneumonia, rest, fluids, and fever reducers.",
                "Early heart attack signs include chest pain/pressure, shortness of breath, cold sweat, nausea, and pain radiating to arm or jaw.",
            ],
            'intent': [
                'medicine_info', 'nutrition_advice', 'general_faq', 'treatment_procedure', 
                'symptoms_diagnosis', 'medicine_info', 'nutrition_advice', 'medicine_info',
                'treatment_procedure', 'symptoms_diagnosis', 'medicine_info', 'nutrition_advice',
                'general_faq', 'treatment_procedure', 'symptoms_diagnosis'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        output_path = self.raw_dir / "sample_data.csv"
        df.to_csv(output_path, index=False)
        return df
    
    def preprocess_for_training(self, df: pd.DataFrame) -> Dict:
        """Preprocess data for model training"""
        logger.info("Preprocessing data for training...")
        
        # Split by intent
        intent_data = {}
        for intent in settings.INTENT_CATEGORIES:
            intent_df = df[df['intent'] == intent]
            intent_data[intent] = {
                'questions': intent_df['question'].tolist(),
                'answers': intent_df['answer'].tolist()
            }
        
        # Save processed data
        output_path = self.processed_dir / "training_data.json"
        with open(output_path, 'w') as f:
            json.dump(intent_data, f, indent=2)
        
        logger.info(f"Processed data saved to {output_path}")
        
        # Create train/val split
        from sklearn.model_selection import train_test_split
        
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['intent'])
        
        train_df.to_csv(self.processed_dir / "train.csv", index=False)
        val_df.to_csv(self.processed_dir / "val.csv", index=False)
        
        logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
        
        return {
            'train': train_df,
            'val': val_df,
            'intent_data': intent_data
        }
    
    def prepare_all_data(self):
        """Main function to prepare all data"""
        logger.info("Starting data preparation pipeline...")
        
        # Try to download real data, fallback to sample
        df = self.download_medquad_dataset()
        
        # Preprocess
        processed = self.preprocess_for_training(df)
        
        logger.info("Data preparation complete!")
        return processed


if __name__ == "__main__":
    prep = DataPreparation()
    prep.prepare_all_data()
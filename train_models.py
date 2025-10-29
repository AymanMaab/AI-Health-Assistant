#!/usr/bin/env python3
"""
Complete training pipeline for AI Health Assistant
Trains both intent classifier and prepares QA engine
"""

import sys
from pathlib import Path

# Add src to path
#sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
import pandas as pd
from src.data_preparation import DataPreparation
from src.intent_classifier import IntentClassifier
from src.qa_engine import QAEngine
from src.config import settings


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("AI HEALTH INFORMATION ASSISTANT - TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Preparation
    print("\n[STEP 1/3] Preparing training data...")
    print("-" * 60)
    data_prep = DataPreparation()
    processed_data = data_prep.prepare_all_data()
    
    train_df = processed_data['train']
    val_df = processed_data['val']
    
    print(f"✓ Training samples: {len(train_df)}")
    print(f"✓ Validation samples: {len(val_df)}")
    print(f"✓ Intent distribution:")
    print(train_df['intent'].value_counts())
    
    # Step 2: Train Intent Classifier
    print("\n[STEP 2/3] Training intent classifier...")
    print("-" * 60)
    print(f"Model: {settings.INTENT_MODEL_NAME}")
    print(f"Device: {settings.DEVICE}")
    print(f"Classes: {len(settings.INTENT_CATEGORIES)}")
    
    classifier = IntentClassifier()
    
    try:
        classifier.train(train_df, val_df)
        print("✓ Intent classifier trained successfully!")
        
        # Test predictions
        test_queries = [
            "What are the side effects of ibuprofen?",
            "How much vitamin D do I need?",
            "What causes headaches?"
        ]
        
        print("\n  Testing predictions:")
        for query in test_queries:
            result = classifier.predict(query)
            print(f"    • '{query}'")
            print(f"      → Intent: {result['intent']} ({result['confidence']:.1%})")
        
    except Exception as e:
        logger.error(f"Intent classifier training failed: {e}")
        print(f"✗ Error: {e}")
        print("  Continuing with pretrained model...")
    
    # Step 3: Setup QA Engine
    print("\n[STEP 3/3] Setting up QA engine...")
    print("-" * 60)
    print(f"QA Model: {settings.QA_MODEL_NAME}")
    
    qa_engine = QAEngine()
    
    try:
        qa_engine.load_models()
        qa_engine.load_knowledge_base()
        print(f"✓ QA engine loaded with {len(qa_engine.knowledge_base)} entries")
        
        # Test QA
        test_question = "What are the side effects of aspirin?"
        print(f"\n  Testing QA:")
        print(f"    Question: {test_question}")
        
        result = qa_engine.answer_question(test_question)
        print(f"    Answer: {result['answer'][:100]}...")
        print(f"    Confidence: {result['confidence']:.1%}")
        print(f"    Source: {result['source']}")
        
    except Exception as e:
        logger.error(f"QA engine setup failed: {e}")
        print(f"✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModels saved to: {settings.MODELS_DIR}")
    print("\nNext steps:")
    print("  1. Start the API server:")
    print("     python src/main.py")
    print("\n  2. Or use Docker:")
    print("     docker-compose up")
    print("\n  3. Test with Postman or cURL")
    print("\n  4. View API docs at:")
    print("     http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
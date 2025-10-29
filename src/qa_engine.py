import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import logging
from typing import List, Dict, Optional, Any
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAEngine:
    """Question Answering using BioBERT, semantic search, and external API fallback."""

    def __init__(self):
        self.qa_model_name = settings.QA_MODEL_NAME
        self.device = settings.DEVICE
        self.qa_pipeline = None
        self.sentence_model = None
        self.knowledge_base: List[Dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None
        
        # Improved thresholds
        self.HIGH_CONFIDENCE_THRESHOLD = 0.70
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.50
        self.LOW_CONFIDENCE_THRESHOLD = 0.30
        self.API_FALLBACK_THRESHOLD = 0.50

    def load_models(self):
        logger.info("Loading QA models...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("BioBERT QA model loaded successfully.")
        except Exception as e:
            logger.warning(f"BioBERT load failed ({e}), using fallback DistilBERT.")
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if self.device == "cuda" else -1
            )

        self.sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.sentence_model.to(self.device)
        logger.info("Models loaded successfully.")

    def load_knowledge_base(self):
        kb_path = settings.PROCESSED_DATA_DIR / "train.csv"
        if kb_path.exists():
            try:
                df = pd.read_csv(kb_path)
                df = df.fillna("").replace([float("inf"), float("-inf")], "")
                if "question" not in df.columns or "answer" not in df.columns:
                    raise ValueError("train.csv missing required 'question' or 'answer' columns")
                self.knowledge_base = [
                    {"question": str(row["question"]).strip(),
                     "answer": str(row["answer"]).strip(),
                     "intent": str(row.get("intent", "")).strip()}
                    for _, row in df.iterrows()
                    if str(row.get("question", "")).strip() and str(row.get("answer", "")).strip()
                ]
                if not self.knowledge_base:
                    raise ValueError("No valid rows found in train.csv")
                logger.info(f"Knowledge base loaded from {kb_path}: {len(self.knowledge_base)} entries.")
            except Exception as e:
                logger.error(f"Failed to load knowledge base ({e}), using fallback.")
                self._load_fallback_kb()
        else:
            logger.warning(f"{kb_path} not found. Using fallback knowledge base.")
            self._load_fallback_kb()

        questions = [q["question"] for q in self.knowledge_base]
        self.embeddings = self.sentence_model.encode(
            questions, convert_to_tensor=True, show_progress_bar=True
        )
        logger.info(f"Knowledge base ready with {len(self.knowledge_base)} entries.")

    def _load_fallback_kb(self):
        """Enhanced fallback knowledge base with comprehensive medical Q&A."""
        self.knowledge_base = [
            {"question": "What is hypertension?",
             "answer": "Hypertension, or high blood pressure, occurs when the force of blood against artery walls is consistently too high (≥130/80 mmHg). It's often called the 'silent killer' because it may have no symptoms but increases risk of heart disease, stroke, and kidney damage.",
             "intent": "general_faq"},
            {"question": "What are the side effects of metformin?",
             "answer": "Common side effects of metformin include nausea, diarrhea, stomach upset, and metallic taste. These usually improve over time. Serious side effects like lactic acidosis are rare but can occur, especially in patients with kidney problems.",
             "intent": "medicine_info"},
            {"question": "How much protein should I eat daily?",
             "answer": "The recommended daily protein intake is around 0.8 grams per kilogram of body weight for adults. Athletes may need 1.2-2.0 g/kg. For a 70kg person, that's about 56 grams per day.",
             "intent": "nutrition_advice"},
            {"question": "Which of the following is the least filtered in glomerules?",
             "answer": "Proteins, especially albumin, are the least filtered in glomerules due to their large size (>70kDa) and negative charge. The glomerular filtration barrier has three layers: fenestrated endothelium, basement membrane, and podocyte filtration slits, which prevent large molecules from passing through. Under normal conditions, <0.1% of plasma proteins are filtered.",
             "intent": "general_faq"},
            {"question": "Which type of Human papilloma virus is most commonly associated with Cervical cancer?",
             "answer": "HPV-16 and HPV-18 are the high-risk types most commonly associated with cervical cancer, accounting for approximately 70% of cases worldwide. HPV-16 alone causes about 50% of cervical cancers and is also linked to oropharyngeal and other cancers.",
             "intent": "general_faq"},
            {"question": "Which vaccine is recommended for adults over 50?",
             "answer": "Adults over 50 should receive: Shingles vaccine (Shingrix, 2 doses), annual influenza vaccine, Tdap booster every 10 years, and pneumococcal vaccines. Adults 65+ also need PPSV23 and PCV15/20 pneumococcal vaccines.",
             "intent": "medicine_info"},
            {"question": "What is the glomerular filtration barrier?",
             "answer": "The glomerular filtration barrier consists of three layers: 1) Fenestrated endothelium (blocks blood cells), 2) Glomerular basement membrane (charge-selective barrier with negative charge), 3) Podocyte filtration slits with slit diaphragms (size-selective barrier). Together they allow passage of water and small solutes while retaining proteins and blood cells.",
             "intent": "general_faq"},
            {"question": "What causes diabetes?",
             "answer": "Diabetes mellitus is caused by insufficient insulin production (Type 1, autoimmune) or insulin resistance (Type 2). Risk factors include genetics, obesity, sedentary lifestyle, age, and family history. It results in chronic hyperglycemia.",
             "intent": "general_faq"},
            {"question": "What is aspirin used for?",
             "answer": "Aspirin (acetylsalicylic acid) is used for pain relief, reducing fever, and reducing inflammation. Low-dose aspirin (75-100mg daily) is commonly used to prevent heart attacks and strokes by preventing blood clots. It works by inhibiting cyclooxygenase (COX) enzymes.",
             "intent": "medicine_info"},
        ]

    def find_relevant_context(self, query: str, intent: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """Find most relevant knowledge base entries."""
        if self.embeddings is None:
            self.load_knowledge_base()

        if intent:
            filtered_kb = [item for item in self.knowledge_base if item.get("intent") == intent]
            if filtered_kb:
                filtered_questions = [item["question"] for item in filtered_kb]
                filtered_embeddings = self.sentence_model.encode(
                    filtered_questions, convert_to_tensor=True
                )
            else:
                filtered_kb, filtered_embeddings = self.knowledge_base, self.embeddings
        else:
            filtered_kb, filtered_embeddings = self.knowledge_base, self.embeddings

        query_embedding = self.sentence_model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, filtered_embeddings)[0]
        cos_scores = torch.nan_to_num(cos_scores, nan=0.0, posinf=1.0, neginf=0.0)

        top_results = torch.topk(cos_scores, k=min(top_k, len(filtered_kb)))
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            results.append({**filtered_kb[int(idx)], "similarity_score": float(score)})
        
        return results

    def answer_question(self, query: str, intent: Optional[str] = None, use_external_api: bool = True) -> Dict:
        """Generate answer with improved accuracy and external API fallback."""
        if self.qa_pipeline is None:
            self.load_models()
        if not self.knowledge_base:
            self.load_knowledge_base()

        relevant_contexts = self.find_relevant_context(query, intent, top_k=5)
        
        if not relevant_contexts:
            if use_external_api:
                return self._try_external_api_fallback(query, intent)
            return {
                "answer": "I couldn't find relevant information in my knowledge base. Please try rephrasing your question or provide more context.",
                "confidence": 0.0,
                "source": "none",
                "contexts": [],
                "used_external_api": False
            }

        top_match = relevant_contexts[0]
        top_score = top_match["similarity_score"]

        logger.info(f"Query: '{query[:50]}...' | Top score: {top_score:.3f} | Intent: {intent}")

        # Strategy 1: High confidence direct match
        if top_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return {
                "answer": top_match["answer"],
                "confidence": float(top_score),
                "source": "knowledge_base",
                "similar_question": top_match["question"],
                "contexts": relevant_contexts[:3],
                "used_external_api": False
            }

        # Strategy 2: Medium confidence - try QA model
        if top_score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            try:
                context = top_match["answer"]
                qa_result = self.qa_pipeline(
                    question=query,
                    context=context,
                    max_answer_len=200,
                    handle_impossible_answer=True
                )
                
                if qa_result.get("score", 0) > 0.1:
                    return {
                        "answer": qa_result["answer"],
                        "confidence": float(qa_result["score"]) * float(top_score),
                        "source": "qa_model",
                        "similar_question": top_match["question"],
                        "contexts": relevant_contexts[:3],
                        "used_external_api": False
                    }
            except Exception as e:
                logger.error(f"QA pipeline error: {e}")

        # Strategy 3: Low confidence - try external APIs FIRST
        if top_score < self.API_FALLBACK_THRESHOLD and use_external_api:
            logger.info(f"Low confidence ({top_score:.2f}), trying external API fallback first...")
            api_result = self._try_external_api_fallback(query, intent)
            if api_result.get("used_external_api") and api_result.get("confidence", 0) > 0.5:
                logger.info(f"✓ External API returned good answer (confidence: {api_result['confidence']})")
                return api_result
            else:
                logger.info("External API didn't return good answer, falling back to knowledge base")

        # Strategy 4: Low-medium confidence - return best match with disclaimer
        if top_score >= self.LOW_CONFIDENCE_THRESHOLD:
            return {
                "answer": f"{top_match['answer']}\n\n⚠️ **Note:** This answer is based on a similar question: '{top_match['question']}'. The information may not fully match your specific query. Please verify or consult a healthcare professional.",
                "confidence": float(top_score),
                "source": "knowledge_base_low_confidence",
                "similar_question": top_match["question"],
                "contexts": relevant_contexts[:3],
                "used_external_api": False
            }

        # Strategy 5: Very low confidence - one more API attempt
        if use_external_api:
            logger.info("Very low confidence, last attempt with external API...")
            api_result = self._try_external_api_fallback(query, intent)
            if api_result.get("used_external_api"):
                return api_result

        # Strategy 6: Give up gracefully
        return {
            "answer": f"❌ **Unable to provide a reliable answer**\n\nI couldn't find accurate information for: '{query}'\n\n**What you can do:**\n1. Rephrase your question more specifically\n2. Try asking about: '{top_match['question']}'\n3. Check if there are typos in medical terms\n4. Consult a healthcare professional for personalized advice\n\n**Closest match found:** {top_match['question']} (confidence: {top_score:.1%})",
            "confidence": float(top_score),
            "source": "no_confident_match",
            "similar_question": top_match["question"],
            "contexts": relevant_contexts[:3],
            "used_external_api": False
        }

    def _try_external_api_fallback(self, query: str, intent: Optional[str]) -> Dict:
        """Try to get answer from external APIs based on intent and query keywords."""
        try:
            # ✅ FIXED IMPORT - Changed from "external_apis" to "src.external_apis"
            from src.external_apis import ExternalAPIManager
            import asyncio
            import re
            
            api_manager = ExternalAPIManager()
            
            query_lower = query.lower()
            words = re.findall(r'\b[a-z]{3,}\b', query_lower)
            
            logger.info(f"Extracted keywords: {words}")
            
            drug_keywords = ['metformin', 'aspirin', 'ibuprofen', 'paracetamol', 'acetaminophen',
                           'warfarin', 'insulin', 'amoxicillin', 'lisinopril', 'atorvastatin',
                           'levothyroxine', 'amlodipine', 'metoprolol', 'omeprazole', 'losartan',
                           'heparin', 'simvastatin', 'gabapentin', 'prednisone', 'tramadol']
            
            food_keywords = ['banana', 'egg', 'fish', 'milk', 'spinach', 'chicken', 'apple', 
                           'orange', 'avocado', 'broccoli', 'salmon', 'beef', 'rice', 'bread',
                           'yogurt', 'cheese', 'potato', 'tomato', 'carrot', 'beans']
            
            # Medicine/Drug queries
            if (intent == "medicine_info" or 
                any(word in query_lower for word in ['drug', 'medicine', 'medication', 'pill', 
                                                      'tablet', 'side effect', 'dose', 'dosage',
                                                      'prescription', 'antibiotic', 'painkiller',
                                                      'used for', 'treats', 'treatment'])):
                
                found_drug = None
                for drug in drug_keywords:
                    if drug in query_lower:
                        found_drug = drug
                        break
                
                if not found_drug:
                    for word in words:
                        if len(word) > 4 and word not in ['question', 'following', 'substance', 'about', 'what']:
                            found_drug = word
                            break
                
                if found_drug:
                    logger.info(f"🔍 Trying OpenFDA API for drug: {found_drug}")
                    try:
                        result = asyncio.run(api_manager.get_medicine_info(found_drug))
                        
                        drug_info = result.get("drug_info", {})
                        if drug_info.get("error"):
                            logger.warning(f"OpenFDA returned error: {drug_info.get('error')}")
                        else:
                            answer_parts = []
                            answer_parts.append(f"**{found_drug.title()} - Drug Information (FDA)**\n")
                            
                            if drug_info.get('drug_name'):
                                answer_parts.append(f"**Brand Name:** {drug_info['drug_name']}")
                            if drug_info.get('generic_name'):
                                answer_parts.append(f"**Generic Name:** {drug_info['generic_name']}")
                            if drug_info.get('manufacturer'):
                                answer_parts.append(f"**Manufacturer:** {drug_info['manufacturer']}\n")
                            
                            if drug_info.get('indications'):
                                answer_parts.append(f"**What it's used for:**\n{drug_info['indications']}\n")
                            if drug_info.get('dosage'):
                                answer_parts.append(f"**Dosage:**\n{drug_info['dosage']}\n")
                            if drug_info.get('warnings'):
                                answer_parts.append(f"**⚠️ Warnings:**\n{drug_info['warnings']}\n")
                            if drug_info.get('adverse_reactions'):
                                answer_parts.append(f"**Side Effects:**\n{drug_info['adverse_reactions']}\n")
                            
                            if len(answer_parts) > 3:  # More than just title and basic info
                                answer = "\n".join(answer_parts)
                                logger.info("✅ OpenFDA API returned good data!")
                                return {
                                    "answer": answer,
                                    "confidence": 0.90,
                                    "source": "external_api_openfda",
                                    "contexts": [],
                                    "used_external_api": True,
                                    "api_data": result
                                }
                    except Exception as e:
                        logger.error(f"OpenFDA API call failed: {e}")
            
            # Nutrition queries
            if (intent == "nutrition_advice" or 
                any(word in query_lower for word in ['food', 'nutrition', 'nutrient', 'vitamin', 
                                                      'protein', 'diet', 'eat', 'calories',
                                                      'fat', 'carb', 'fiber', 'mineral'])):
                
                found_food = None
                for food in food_keywords:
                    if food in query_lower:
                        found_food = food
                        break
                
                if not found_food:
                    for word in words:
                        if len(word) > 3 and word not in ['question', 'following', 'nutrition', 'what', 'about']:
                            found_food = word
                            break
                
                if found_food:
                    logger.info(f"🔍 Trying USDA API for food: {found_food}")
                    try:
                        result = asyncio.run(api_manager.get_nutrition_info(found_food))
                        
                        nutrition = result.get("nutrition_data", {})
                        if nutrition.get("error"):
                            logger.warning(f"USDA returned error: {nutrition.get('error')}")
                        else:
                            answer_parts = []
                            answer_parts.append(f"**{found_food.title()} - Nutrition Information (USDA)**\n")
                            answer_parts.append(f"**Food:** {nutrition.get('food_name', found_food)}")
                            answer_parts.append(f"**Serving Size:** {nutrition.get('serving_size')} {nutrition.get('serving_unit')}\n")
                            answer_parts.append(f"**Key Nutrients:**")
                            
                            nutrients = nutrition.get('nutrients', {})
                            if nutrients:
                                for nutrient, value in list(nutrients.items())[:10]:
                                    answer_parts.append(f"  • {nutrient}: {value}")
                                
                                answer = "\n".join(answer_parts)
                                logger.info("✅ USDA API returned good data!")
                                return {
                                    "answer": answer,
                                    "confidence": 0.85,
                                    "source": "external_api_usda",
                                    "contexts": [],
                                    "used_external_api": True,
                                    "api_data": result
                                }
                    except Exception as e:
                        logger.error(f"USDA API call failed: {e}")
            
        except Exception as e:
            logger.error(f"External API fallback error: {e}", exc_info=True)
        
        return {
            "answer": "I couldn't find a reliable answer in my knowledge base or external sources.",
            "confidence": 0.0,
            "source": "api_fallback_failed",
            "contexts": [],
            "used_external_api": False
        }

    def batch_answer(self, queries: List[str], use_external_api: bool = True) -> List[Dict]:
        """Process multiple queries in batch."""
        results = []
        for q in queries:
            try:
                results.append(self.answer_question(q, use_external_api=use_external_api))
            except Exception as e:
                logger.error(f"Error processing '{q}': {e}")
                results.append({
                    "answer": "Error processing this question.",
                    "confidence": 0.0,
                    "source": "error",
                    "contexts": [],
                    "used_external_api": False
                })
        return results

    def add_to_knowledge_base(self, question: str, answer: str, intent: str = "") -> bool:
        """Dynamically add new Q&A pairs to the knowledge base."""
        try:
            new_entry = {
                "question": question.strip(),
                "answer": answer.strip(),
                "intent": intent.strip()
            }
            self.knowledge_base.append(new_entry)
            
            questions = [q["question"] for q in self.knowledge_base]
            self.embeddings = self.sentence_model.encode(
                questions, convert_to_tensor=True
            )
            
            logger.info(f"Added new entry to knowledge base: {question}")
            return True
        except Exception as e:
            logger.error(f"Failed to add to knowledge base: {e}")
            return False


if __name__ == "__main__":
    engine = QAEngine()
    engine.load_models()
    engine.load_knowledge_base()

    test_queries = [
        "What are the side effects of metformin?",
        "Which of the following is the least filtered in glomerules?",
        "Which type of HPV causes cervical cancer?",
        "What vaccines do adults over 50 need?",
        "How much protein should I eat?",
        "What is aspirin used for?",
        "What nutrients are in banana?",
    ]

    print("\n" + "="*80)
    print("QA ENGINE TEST WITH EXTERNAL API FALLBACK")
    print("="*80)

    for query in test_queries:
        result = engine.answer_question(query, use_external_api=True)
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print(f"{'─'*80}")
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Source: {result['source']}")
        print(f"Used External API: {result.get('used_external_api', False)}")
        print(f"{'─'*80}")
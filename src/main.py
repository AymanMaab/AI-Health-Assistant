# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from src.config import settings
from src.intent_classifier import IntentClassifier
from src.qa_engine import QAEngine
from src.external_apis import ExternalAPIManager

# -----------------------------
# Setup logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Global instances (lazy loading)
# -----------------------------
intent_classifier: Optional[IntentClassifier] = None
qa_engine: Optional[QAEngine] = None
api_manager: Optional[ExternalAPIManager] = None

# -----------------------------
# Lifespan for startup/shutdown
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    logger.info("Starting AI Health Information Assistant...")
    logger.info(f"Device: {settings.DEVICE}")

    # Load models
    try:
        get_intent_classifier()
        logger.info("Intent classifier loaded")

        get_qa_engine()
        logger.info("QA engine loaded")

        get_api_manager()
        logger.info("API manager initialized")

        logger.info("All models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)

    yield  # Application running

    # Shutdown
    logger.info("Shutting down AI Health Information Assistant...")

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered healthcare information assistant with NLP capabilities",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Pydantic models
# -----------------------------
class HealthQuery(BaseModel):
    question: str = Field(..., min_length=5, max_length=500, description="Health-related question")
    use_external_api: bool = Field(default=True, description="Whether to use external APIs")

    class Config:
        schema_extra = {
            "example": {
                "question": "What are the side effects of metformin?",
                "use_external_api": True
            }
        }

class MedicineQuery(BaseModel):
    medicine_name: str = Field(..., min_length=2, max_length=100)

    class Config:
        schema_extra = {
            "example": {
                "medicine_name": "aspirin"
            }
        }

class NutritionQuery(BaseModel):
    food_name: str = Field(..., min_length=2, max_length=100)

    class Config:
        schema_extra = {
            "example": {
                "food_name": "banana"
            }
        }

class HealthResponse(BaseModel):
    question: str
    answer: str
    intent: str
    confidence: float
    source: str
    timestamp: str
    external_data: Optional[Dict] = None
    similar_questions: Optional[List[str]] = None

# -----------------------------
# Helper functions for lazy loading
# -----------------------------
def get_intent_classifier():
    global intent_classifier
    if intent_classifier is None:
        intent_classifier = IntentClassifier()
        intent_classifier.load()
    return intent_classifier

def get_qa_engine():
    global qa_engine
    if qa_engine is None:
        qa_engine = QAEngine()
        qa_engine.load_models()
        qa_engine.load_knowledge_base()
    return qa_engine

def get_api_manager():
    global api_manager
    if api_manager is None:
        api_manager = ExternalAPIManager()
    return api_manager

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "Welcome to AI Health Information Assistant API",
        "version": settings.VERSION,
        "endpoints": {
            "ask": f"{settings.API_PREFIX}/ask",
            "medicine_info": f"{settings.API_PREFIX}/medicine-info",
            "nutrition": f"{settings.API_PREFIX}/nutrition",
            "health_check": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "intent_classifier": intent_classifier is not None,
            "qa_engine": qa_engine is not None
        }
    }

@app.post(f"{settings.API_PREFIX}/ask", response_model=HealthResponse)
async def ask_question(query: HealthQuery):
    try:
        logger.info(f"Received query: {query.question}")
        classifier = get_intent_classifier()
        qa = get_qa_engine()

        # Intent classification
        intent_result = classifier.predict(query.question)
        intent = intent_result["intent"]
        intent_confidence = intent_result["confidence"]
        logger.info(f"Intent: {intent} (confidence: {intent_confidence:.2%})")

        # Generate answer
        qa_result = qa.answer_question(query.question, intent=intent, use_external_api=query.use_external_api)

        response = {
            "question": query.question,
            "answer": qa_result["answer"],
            "intent": intent,
            "confidence": qa_result["confidence"],
            "source": qa_result["source"],
            "timestamp": datetime.now().isoformat(),
            "external_data": qa_result.get("api_data") if qa_result.get("used_external_api") else None,
            "similar_questions": [ctx["question"] for ctx in qa_result.get("contexts", [])[:3]] if qa_result.get("contexts") else None
        }

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_PREFIX}/medicine-info")
async def get_medicine_info(query: MedicineQuery):
    try:
        logger.info(f"Medicine info request: {query.medicine_name}")
        api_mgr = get_api_manager()
        result = await api_mgr.get_medicine_info(query.medicine_name)
        return {"medicine": query.medicine_name, "data": result, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error fetching medicine info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_PREFIX}/nutrition")
async def get_nutrition_info(query: NutritionQuery):
    try:
        logger.info(f"Nutrition info request: {query.food_name}")
        api_mgr = get_api_manager()
        result = await api_mgr.get_nutrition_info(query.food_name)
        return {"food": query.food_name, "data": result, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error fetching nutrition info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_PREFIX}/intents")
async def list_intents():
    return {
        "intents": settings.INTENT_CATEGORIES,
        "descriptions": {
            "medicine_info": "Questions about medications, drugs, dosages",
            "nutrition_advice": "Questions about diet, nutrition, foods",
            "general_faq": "General health questions",
            "symptoms_diagnosis": "Questions about symptoms and diagnosis",
            "treatment_procedure": "Questions about treatments and procedures"
        }
    }

@app.post(f"{settings.API_PREFIX}/batch")
async def batch_questions(queries: List[HealthQuery]):
    if len(queries) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 queries per batch")

    results = []
    qa = get_qa_engine()
    classifier = get_intent_classifier()

    for query in queries:
        try:
            intent_result = classifier.predict(query.question)
            intent = intent_result["intent"]
            qa_result = qa.answer_question(query.question, intent=intent, use_external_api=query.use_external_api)

            results.append({
                "question": query.question,
                "answer": qa_result["answer"],
                "intent": intent,
                "confidence": qa_result["confidence"],
                "source": qa_result["source"],
                "timestamp": datetime.now().isoformat(),
                "similar_questions": [ctx["question"] for ctx in qa_result.get("contexts", [])[:3]] if qa_result.get("contexts") else None
            })
        except Exception as e:
            logger.error(f"Error processing query '{query.question}': {e}", exc_info=True)
            results.append({
                "question": query.question,
                "answer": "Error processing this question",
                "intent": "error",
                "confidence": 0.0,
                "source": "error",
                "timestamp": datetime.now().isoformat()
            })

    return {"count": len(results), "results": results, "timestamp": datetime.now().isoformat()}

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.RELOAD)

import httpx
import asyncio
from typing import Dict, List, Optional
import logging
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenFDAAPI:
    """Interface to OpenFDA API for drug information"""
    
    BASE_URL = "https://api.fda.gov/drug"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENFDA_API_KEY
        
    async def search_drug(self, drug_name: str) -> Dict:
        """Search for drug information"""
        try:
            endpoint = f"{self.BASE_URL}/label.json"
            params = {
                "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
                "limit": 1
            }
            
            # Only add API key if it exists and is not empty
            if self.api_key and self.api_key.strip():
                params["api_key"] = self.api_key
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(endpoint, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("results"):
                        return self._parse_drug_label(data["results"][0])
                    else:
                        return {"error": "Drug not found", "status": "not_found"}
                elif response.status_code == 403:
                    logger.warning(f"OpenFDA API key rejected (403). Using without key or check your key.")
                    # Try again without API key
                    params.pop("api_key", None)
                    retry_response = await client.get(endpoint, params=params)
                    if retry_response.status_code == 200:
                        data = retry_response.json()
                        if data.get("results"):
                            return self._parse_drug_label(data["results"][0])
                    return {"error": f"API access forbidden. Check your API key.", "status": "forbidden"}
                else:
                    return {"error": f"API error: {response.status_code}", "status": "error"}
                    
        except Exception as e:
            logger.error(f"OpenFDA API error: {e}")
            return {"error": str(e), "status": "exception"}
    
    def _parse_drug_label(self, label_data: Dict) -> Dict:
        """Parse FDA drug label data"""
        try:
            openfda = label_data.get("openfda", {})
            
            return {
                "drug_name": openfda.get("brand_name", ["Unknown"])[0] if openfda.get("brand_name") else "Unknown",
                "generic_name": openfda.get("generic_name", ["Unknown"])[0] if openfda.get("generic_name") else "Unknown",
                "manufacturer": openfda.get("manufacturer_name", ["Unknown"])[0] if openfda.get("manufacturer_name") else "Unknown",
                "indications": label_data.get("indications_and_usage", ["Not available"])[0][:500] if label_data.get("indications_and_usage") else "Not available",
                "dosage": label_data.get("dosage_and_administration", ["Not available"])[0][:500] if label_data.get("dosage_and_administration") else "Not available",
                "warnings": label_data.get("warnings", ["Not available"])[0][:500] if label_data.get("warnings") else "Not available",
                "adverse_reactions": label_data.get("adverse_reactions", ["Not available"])[0][:500] if label_data.get("adverse_reactions") else "Not available",
                "drug_interactions": label_data.get("drug_interactions", ["Not available"])[0][:500] if label_data.get("drug_interactions") else "Not available",
            }
        except Exception as e:
            logger.error(f"Error parsing drug label: {e}")
            return {"error": "Failed to parse drug information"}
    
    async def get_adverse_events(self, drug_name: str, limit: int = 5) -> Dict:
        """Get adverse events for a drug"""
        try:
            endpoint = f"{self.BASE_URL}/event.json"
            params = {
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": limit
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(endpoint, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "drug_name": drug_name,
                        "top_reactions": [
                            {
                                "reaction": item["term"],
                                "count": item["count"]
                            }
                            for item in data.get("results", [])[:limit]
                        ]
                    }
                else:
                    return {"error": f"API error: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"OpenFDA adverse events error: {e}")
            return {"error": str(e)}


class NutritionAPI:
    """Interface for nutrition data"""
    
    BASE_URL = "https://api.nal.usda.gov/fdc/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        # Using demo key for testing
        self.api_key = api_key or "DEMO_KEY"
    
    async def search_food(self, food_name: str) -> Dict:
        """Search for food nutrition information"""
        try:
            endpoint = f"{self.BASE_URL}/foods/search"
            params = {
                "api_key": self.api_key,
                "query": food_name,
                "pageSize": 1
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(endpoint, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("foods"):
                        return self._parse_nutrition_data(data["foods"][0])
                    else:
                        return {"error": "Food not found"}
                else:
                    return {"error": f"API error: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Nutrition API error: {e}")
            return {"error": str(e)}
    
    def _parse_nutrition_data(self, food_data: Dict) -> Dict:
        """Parse nutrition data"""
        try:
            nutrients = {}
            for nutrient in food_data.get("foodNutrients", [])[:10]:
                name = nutrient.get("nutrientName", "Unknown")
                value = nutrient.get("value", 0)
                unit = nutrient.get("unitName", "")
                nutrients[name] = f"{value} {unit}"
            
            return {
                "food_name": food_data.get("description", "Unknown"),
                "brand": food_data.get("brandOwner", "Generic"),
                "nutrients": nutrients,
                "serving_size": food_data.get("servingSize", "N/A"),
                "serving_unit": food_data.get("servingSizeUnit", "")
            }
        except Exception as e:
            logger.error(f"Error parsing nutrition data: {e}")
            return {"error": "Failed to parse nutrition information"}


class ExternalAPIManager:
    """Manages all external API calls"""
    
    def __init__(self):
        self.fda_api = OpenFDAAPI()
        self.nutrition_api = NutritionAPI()
    
    async def get_medicine_info(self, medicine_name: str) -> Dict:
        """Get comprehensive medicine information"""
        drug_info = await self.fda_api.search_drug(medicine_name)
        adverse_events = await self.fda_api.get_adverse_events(medicine_name)
        
        return {
            "medicine": medicine_name,
            "drug_info": drug_info,
            "adverse_events": adverse_events,
            "source": "OpenFDA"
        }
    
    async def get_nutrition_info(self, food_name: str) -> Dict:
        """Get nutrition information"""
        nutrition_data = await self.nutrition_api.search_food(food_name)
        
        return {
            "food": food_name,
            "nutrition_data": nutrition_data,
            "source": "USDA FoodData Central"
        }
    
    async def enhanced_query(self, query: str, intent: str) -> Optional[Dict]:
        """Enhance answer with external API data"""
        try:
            if intent == "medicine_info":
                # Extract medicine name (simple approach)
                words = query.lower().split()
                # Common medicine-related words to skip
                skip_words = {'what', 'are', 'the', 'side', 'effects', 'of', 'about', 'is', 'does', 
                             'how', 'can', 'medicine', 'drug', 'medication', 'pill', 'tablet'}
                
                for word in words:
                    if len(word) > 3 and word not in skip_words:
                        result = await self.get_medicine_info(word)
                        if "error" not in result.get("drug_info", {}):
                            return result
            
            elif intent == "nutrition_advice":
                # Extract food name
                words = query.lower().split()
                skip_words = {'what', 'are', 'the', 'nutrition', 'much', 'many', 'is', 'in', 
                             'about', 'how', 'food', 'eat', 'eating'}
                
                for word in words:
                    if len(word) > 3 and word not in skip_words:
                        result = await self.get_nutrition_info(word)
                        if "error" not in result.get("nutrition_data", {}):
                            return result
            
            return None
            
        except Exception as e:
            logger.error(f"Enhanced query error: {e}")
            return None


# Test functions
async def test_apis():
    """Test external APIs"""
    manager = ExternalAPIManager()
    
    # Test medicine info
    print("\n" + "="*60)
    print("Testing Medicine API (OpenFDA)")
    print("="*60)
    medicine_result = await manager.get_medicine_info("aspirin")
    print(f"Medicine: {medicine_result.get('medicine')}")
    print(f"Drug Info: {medicine_result.get('drug_info')}")
    print(f"Adverse Events: {medicine_result.get('adverse_events')}")
    
    # Test nutrition info
    print("\n" + "="*60)
    print("Testing Nutrition API (USDA)")
    print("="*60)
    nutrition_result = await manager.get_nutrition_info("banana")
    print(f"Food: {nutrition_result.get('food')}")
    print(f"Nutrition Data: {nutrition_result.get('nutrition_data')}")
    
    # Test enhanced query
    print("\n" + "="*60)
    print("Testing Enhanced Query")
    print("="*60)
    enhanced_result = await manager.enhanced_query(
        "What are the side effects of ibuprofen?",
        "medicine_info"
    )
    print(f"Enhanced Result: {enhanced_result}")


if __name__ == "__main__":
    print("Starting External API Tests...")
    print("Note: These APIs may have rate limits or require internet connection")
    asyncio.run(test_apis())
    print("\nTests completed!")
# services/ai_service.py (FINAL CLOUD-READY VERSION)

import os
import logging
import json
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any
import time
import re
import tempfile

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptQualityChecker:
    """Analyzes the generated prompts for common issues."""
    def __init__(self):
        self.generic_terms = ["beautiful", "amazing", "stunning", "gorgeous", "nice", "good", "wonderful"]
        self.cliche_phrases = ["eyes focused on", "slight frown", "rich colors", "deep shadows", "cinematic shot"]

    def check_quality(self, creative_brief: str, technical_prompt: str) -> Dict[str, Any]:
        issues, score = [], 100
        
        generic_found = [term for term in self.generic_terms if term.lower() in technical_prompt.lower()]
        if generic_found:
            issues.append(f"Generic terms found: {', '.join(generic_found)}")
            score -= len(generic_found) * 5

        cliche_found = [phrase for phrase in self.cliche_phrases if phrase.lower() in technical_prompt.lower()]
        if cliche_found:
            issues.append(f"Cliché phrases found: {', '.join(cliche_found)}")
            score -= len(cliche_found) * 10

        if len(creative_brief.split()) < 100:
            issues.append("Creative brief is too short.")
            score -= 15

        if len(technical_prompt.split()) < 80:
            issues.append("Technical prompt is too short.")
            score -= 20
            
        return {'score': max(0, score), 'issues': issues}

class AIService:
    """Handles all interactions with Google Vertex AI services."""
    def __init__(self):
        self.project_id = None
        self.location = "us-central1"
        
        # --- FIX 1: UPDATED MODELS ---
        self.generative_model_name = "gemini-2.5-pro"  # Using a recent, stable model
        self.embedding_model_name = "text-embedding-005" # CRITICAL FIX: Using a current, non-retired model
        
        self.generative_model = None
        self.embedding_model = None
        self.is_configured = False
        self.quality_checker = PromptQualityChecker()
        self.safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }
        self.configure_models()

    # --- FIX 2: CLOUD AUTHENTICATION LOGIC ---
    def _configure_credentials_for_cloud(self) -> Optional[str]:
        """Checks for and configures GCP credentials from Streamlit secrets."""
        if 'gcp_service_account' in st.secrets:
            try:
                creds_json_str = st.secrets["gcp_service_account"]
                creds_dict = json.loads(creds_json_str)
                
                # Create a temporary file to store credentials for the SDK
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_creds_file:
                    json.dump(creds_dict, temp_creds_file)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_file.name
                
                # Return the project_id from the credentials
                return creds_dict.get("project_id")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to process Streamlit secrets for GCP: {e}")
        return None

    def configure_models(self) -> bool:
        """Initializes Vertex AI models, handling both local and cloud environments."""
        if self.is_configured:
            return True
        
        try:
            # Try to configure credentials from secrets (for cloud)
            cloud_project_id = self._configure_credentials_for_cloud()
            
            # Use cloud project ID if available, otherwise fall back to local environment variable
            self.project_id = cloud_project_id or os.getenv("GOOGLE_CLOUD_PROJECT")

            if not self.project_id:
                logger.critical("Google Cloud project ID is not configured.")
                st.error("GCP Project ID not found. Please set it in secrets or as an environment variable.")
                return False

            vertexai.init(project=self.project_id, location=self.location)
            
            self.generative_model = GenerativeModel(self.generative_model_name)
            self.embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model_name)
            
            self.is_configured = True
            logger.info(f"Vertex AI models configured successfully in project '{self.project_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to configure Vertex AI models: {e}", exc_info=True)
            st.error(f"Could not initialize AI Service. Check the application logs for details. Error: {e}")
            self.is_configured = False
            return False

    def load_helios_master_prompt(self) -> str:
        """Loads the main system prompt from a file with a fallback."""
        try:
            # Path relative to this file's location
            helios_path = os.path.join(os.path.dirname(__file__), '..', 'helios_master_prompt.txt')
            if os.path.exists(helios_path):
                with open(helios_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            logger.warning(f"Master prompt not found at {helios_path}, using fallback.")
            return self.get_fallback_prompt()
        except Exception as e:
            logger.error(f"Error loading Helios master prompt: {e}")
            return self.get_fallback_prompt()

    def get_fallback_prompt(self) -> str:
        """Returns a hardcoded fallback system prompt."""
        return """You are Helios, an elite creative director...
CRITICAL: Your entire response MUST be a single, valid JSON object..."""

    def build_enhanced_user_content(self, director_brief: str, retrieved_context: str) -> str:
        """Constructs the final prompt content for the AI model."""
        context_section = f"RELEVANT KNOWLEDGE BASE:\n'''\n{retrieved_context}\n'''\n\n" if retrieved_context else ""
        return f"{context_section}USER'S SCENE DIRECTOR BRIEF:\n'''\n{director_brief.strip()}\n'''"

    def get_ai_response(self, director_brief: str, max_retries: int = 2) -> Dict[str, Any]:
        """Generates a response from the AI, with retries."""
        if not self.is_configured:
            return {"creative_brief": "AI service is not configured.", "technical_prompt": "Config error.", "quality_report": None}

        helios_system_prompt = self.load_helios_master_prompt()
        retrieved_context = self.find_relevant_chunks(director_brief, top_k=5)
        user_content = self.build_enhanced_user_content(director_brief, retrieved_context)
        
        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.8,
            "max_output_tokens": 2048
        }
        
        model_with_system_prompt = GenerativeModel(
            self.generative_model_name,
            system_instruction=helios_system_prompt
        )

        for attempt in range(max_retries):
            try:
                response = model_with_system_prompt.generate_content(
                    [Part.from_text(user_content)],
                    generation_config=generation_config,
                    safety_settings=self.safety_settings
                )
                return self.process_ai_response(response.text)
            except Exception as e:
                logger.error(f"AI generation error (attempt {attempt + 1}): {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return {"creative_brief": "Error: Failed to generate a valid response from the AI.", "technical_prompt": "Please try again or simplify your request.", "quality_report": None}

    def process_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parses the JSON response from the AI and runs a quality check."""
        try:
            # Use regex to find the JSON object in the response text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                response_dict = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON object found in response.", response_text, 0)
            
            brief = response_dict.get("creative_brief", "")
            prompt = response_dict.get("technical_prompt", "")
            quality_report = self.quality_checker.check_quality(brief, prompt)
            response_dict['quality_report'] = quality_report
            return response_dict
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nResponse text: {response_text}")
            return {"creative_brief": "Error parsing AI response.", "technical_prompt": "Received invalid JSON.", "quality_report": None}

    @st.cache_data(ttl=3600)
    def load_vector_db(_self, db_path: str = "vector_database.json") -> List[Dict]:
        """Loads the vector database from a JSON file, cached for performance."""
        try:
            if os.path.exists(db_path):
                with open(db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
        return []

    @st.cache_data(ttl=1800)
    def get_embedding(_self, text: str) -> Optional[List[float]]:
        """Generates embeddings for a given text, cached for performance."""
        if not _self.is_configured or not text or not text.strip():
            return None
        try:
            embeddings = _self.embedding_model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Could not generate embedding for text: '{text[:50]}...'. Error: {e}")
            return None

    def find_relevant_chunks(self, query: str, top_k: int = 5) -> str:
        """Finds the most relevant text chunks from the vector DB based on cosine similarity."""
        vector_db = self.load_vector_db()
        if not vector_db or not query.strip():
            return ""
        
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return ""
            
        query_vec = np.array(query_embedding)
        scores = []
        for chunk in vector_db:
            if "embedding" in chunk and chunk["embedding"]:
                chunk_vec = np.array(chunk['embedding'])
                # Calculate cosine similarity
                similarity = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
                scores.append((similarity, chunk.get('text', '')))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return "\n---\n".join([text for score, text in scores[:top_k] if text.strip()])
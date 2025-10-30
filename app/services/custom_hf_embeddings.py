import requests
import json
import time
from typing import List
from langchain_core.embeddings import Embeddings
from app.config import logger


class CustomHuggingFaceEmbeddings(Embeddings):
    """Custom embeddings class for HuggingFace inference endpoint with retry logic."""

    def __init__(self, endpoint_url: str, api_token: str, max_retries: int = 3, timeout: int = 30):
        self.endpoint_url = endpoint_url
        self.api_token = api_token
        self.max_retries = max_retries
        self.timeout = timeout

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with retry logic."""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding_with_retry(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query with retry logic."""
        return self._get_embedding_with_retry(text)

    def _get_embedding_with_retry(self, text: str) -> List[float]:
        """Get embedding for a single text with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": text
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract embedding from response (format depends on your model)
                    if isinstance(result, list) and len(result) > 0:
                        return result[0]["embedding"] if "embedding" in result[0] else result[0]
                    else:
                        return result
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Error calling HuggingFace endpoint: {response.status_code} - {response.text}")
                    if attempt == self.max_retries - 1:  # Last attempt
                        raise Exception(f"Error calling HuggingFace endpoint: {response.status_code} - {response.text}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                logger.error(f"Request exception on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:  # Last attempt
                    raise Exception(f"Failed to get embedding after {self.max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("Failed to get embedding after max retries")
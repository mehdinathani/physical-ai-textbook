"""
The Tutor: Chat API for the Physical AI & Humanoid Robotics Textbook.

This service implements RAG logic: Embed query -> Search Qdrant -> Construct System Prompt -> Call Google Gemini Chat.
"""

import os
import hashlib
import time
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()

# Simple in-memory cache with TTL
class SimpleCache:
    def __init__(self, ttl: int = 3600):  # TTL in seconds (default 1 hour)
        self.cache: Dict[str, tuple] = {}  # key -> (value, timestamp)
        self.ttl = ttl

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                # Remove expired entry
                del self.cache[key]
        return None

    def set(self, key: str, value: str):
        self.cache[key] = (value, time.time())

    def clear_expired(self):
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

# Initialize cache
query_cache = SimpleCache(ttl=3600)  # 1 hour TTL

# Pydantic model for the request (as specified in the task)
class ChatRequest(BaseModel):
    message: str


# Initialize FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook Chat API",
    description="RAG-powered chat API for the textbook",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text using Google's text-embedding-004 model."""
    # Get API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    # Configure the API key
    genai.configure(api_key=google_api_key)

    # Get the embedding
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=[text],
        task_type="RETRIEVAL_QUERY"  # Using retrieval query as this is for RAG
    )

    return result['embedding'][0]


@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Chat endpoint that implements RAG logic:
    1. Check cache for existing response
    2. If not cached: Embed query
    3. Search Qdrant
    4. Construct system prompt
    5. Call Google Gemini Chat
    6. Cache response
    7. Return answer
    """
    start_time = time.time()
    logger.info(f"Processing chat request: {request.message[:50]}...")

    # Create cache key from the request message
    cache_key = hashlib.sha256(request.message.encode()).hexdigest()

    # Check if response is already cached
    cached_response = query_cache.get(cache_key)
    if cached_response:
        response_time = time.time() - start_time
        logger.info(f"Cache hit for query, response time: {response_time:.2f}s")
        return {"answer": cached_response}

    # Get API keys from environment
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not all([qdrant_url, qdrant_api_key, google_api_key]):
        logger.error("Missing required environment variables")
        return {
            "error": "Missing required environment variables",
            "details": "Please ensure QDRANT_URL, QDRANT_API_KEY, and GOOGLE_API_KEY are set in your environment"
        }

    # Initialize clients
    try:
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=True
        )
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {str(e)}")
        return {
            "error": "Failed to connect to Qdrant",
            "details": f"Please check your QDRANT_URL and QDRANT_API_KEY: {str(e)}"
        }

    # Configure Google API
    try:
        genai.configure(api_key=google_api_key)
    except Exception as e:
        logger.error(f"Failed to configure Google API: {str(e)}")
        return {
            "error": "Failed to configure Google API",
            "details": f"Please check your GOOGLE_API_KEY: {str(e)}"
        }

    try:
        # Step 1: Embed the query
        embedding_start = time.time()
        query_embedding = get_embedding(request.message)
        embedding_time = time.time() - embedding_start
        logger.info(f"Query embedding completed in {embedding_time:.2f}s")

        # Step 2: Search Qdrant for relevant content
        search_start = time.time()
        try:
            search_results = qdrant_client.search(
                collection_name="textbook_docs",
                query_vector=query_embedding,
                limit=5,  # Get top 5 most relevant chunks
                with_payload=True
            )
        except Exception as e:
            search_time = time.time() - search_start
            logger.error(f"Qdrant search failed after {search_time:.2f}s: {str(e)}")
            return {
                "error": "Failed to search Qdrant database",
                "details": f"The Qdrant collection 'textbook_docs' may not exist or be populated. Error: {str(e)}"
            }
        search_time = time.time() - search_start
        logger.info(f"Qdrant search completed in {search_time:.2f}s")

        # Step 3: Construct context from search results
        if not search_results:
            logger.warning("No relevant content found for query")
            return {
                "error": "No relevant content found",
                "details": "The query did not match any content in the textbook database. The ingestion process may need to be run first."
            }

        context_parts = []
        for result in search_results:
            payload = result.payload
            context_parts.append(f"Source: {payload.get('source', 'Unknown')}\nContent: {payload.get('text', '')}")

        context = "\n\n".join(context_parts)

        # Step 4: Configure the Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Step 5: Construct system prompt with context and call Google Gemini
        full_prompt = f"""
        You are an expert tutor for the Physical AI & Humanoid Robotics Textbook.
        Use the following context from the textbook to answer the user's question.
        If the context doesn't contain relevant information, say so.

        Context:
        {context}

        User's question: {request.message}
        """

        completion_start = time.time()
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500
                )
            )
        except Exception as e:
            completion_time = time.time() - completion_start
            logger.error(f"Google Gemini completion failed after {completion_time:.2f}s: {str(e)}")
            return {
                "error": "Google Gemini API error",
                "details": f"Failed to generate response: {str(e)}"
            }
        completion_time = time.time() - completion_start
        logger.info(f"Google Gemini completion completed in {completion_time:.2f}s")

        # Step 6: Return the answer
        if response.candidates and len(response.candidates) > 0:
            answer = response.candidates[0].content.parts[0].text
        else:
            answer = "No response generated by the model."

        # Step 7: Cache the response for future queries
        query_cache.set(cache_key, answer)

        total_time = time.time() - start_time
        logger.info(f"Request completed successfully in {total_time:.2f}s")

        return {"answer": answer}

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Request failed after {total_time:.2f}s: {str(e)}")
        return {
            "error": "An unexpected error occurred",
            "details": str(e)
        }


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"message": "Physical AI & Humanoid Robotics Textbook Chat API is running"}


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Textbook Chat API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
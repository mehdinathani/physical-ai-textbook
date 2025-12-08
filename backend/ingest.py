"""
The Librarian: Ingestion script for the Physical AI & Humanoid Robotics Textbook.

This script:
1. Recursively finds all .md files in frontend/docs
2. Strips Docusaurus frontmatter
3. Chunks text into 500-token chunks with overlap
4. Connects to Qdrant Cloud and uploads points
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
import google.generativeai as genai


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not all([qdrant_url, qdrant_api_key, google_api_key]):
        raise ValueError("Missing required environment variables in .env file")

    return qdrant_url, qdrant_api_key, google_api_key


def find_markdown_files(docs_dir: str) -> List[Path]:
    """Recursively find all .md files in the specified directory."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory {docs_dir} does not exist")

    md_files = list(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")
    return md_files


def strip_frontmatter(content: str) -> str:
    """Strip Docusaurus frontmatter (metadata between --- markers at the top)."""
    # Pattern to match frontmatter: lines between --- at the start of the file
    frontmatter_pattern = r'^---\n.*?\n---\n'
    content_without_frontmatter = re.sub(frontmatter_pattern, '', content, count=1, flags=re.DOTALL)
    return content_without_frontmatter.strip()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Split text into chunks of specified size with overlap."""
    # Simple text splitter that handles markdown content
    chunks = []
    start = 0

    while start < len(text):
        # Determine the end position
        end = start + chunk_size

        # If we're near the end of the text, just take the remainder
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to find a good breaking point (preferably at sentence or paragraph boundaries)
        chunk = text[start:end]

        # Look for a good break point before the end of the chunk
        break_points = ['\n\n', '\n', '. ', '! ', '? ', '; ', ': ']
        break_point_found = False

        for bp in break_points:
            # Look in the last part of the chunk to find a break point
            look_start = max(0, chunk_size - 100)  # Look in the last 100 characters
            reversed_chunk = chunk[look_start:]
            last_pos = reversed_chunk.rfind(bp)

            if last_pos != -1:
                actual_break_pos = look_start + last_pos + len(bp)
                end = start + actual_break_pos
                chunks.append(text[start:end])
                start = end - chunk_overlap  # Overlap
                break_point_found = True
                break

        # If no good break point was found, just take the chunk as is
        if not break_point_found:
            chunks.append(text[start:end])
            start = end - chunk_overlap  # Overlap

        # Ensure we're making progress to avoid infinite loops
        if start <= end - chunk_overlap and not break_point_found:
            start = end  # Move to next position if we can't find a good break point

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text using Google's text-embedding-004 model."""
    # Get API key from environment (assuming it's already configured globally)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    # Configure the API key
    genai.configure(api_key=google_api_key)

    # Get the embedding
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=[text],
        task_type="RETRIEVAL_DOCUMENT"  # Using retrieval document as this is for document chunks
    )

    return result['embedding'][0]


def main():
    """Main ingestion function."""
    print("Starting ingestion process...")

    # Load environment variables
    qdrant_url, qdrant_api_key, google_api_key = load_environment()

    # Initialize clients
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True
    )

    # Configure Google API
    genai.configure(api_key=google_api_key)

    # Set up collection - CRITICAL: Google's text-embedding-004 outputs 768 dimensions
    collection_name = "textbook_docs"
    vector_size = 768  # Google text-embedding-004 embedding dimension

    # Create collection or recreate if dimensions don't match
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        # Check if the vector size matches
        if collection_info.config.params.vectors.size != vector_size:
            print(f"Collection '{collection_name}' exists but has wrong vector size ({collection_info.config.params.vectors.size}). Deleting and recreating...")
            qdrant_client.delete_collection(collection_name)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{collection_name}' recreated with {vector_size}-dimension vectors")
        else:
            print(f"Collection '{collection_name}' already exists with correct vector size ({vector_size})")
    except:
        print(f"Creating collection '{collection_name}'")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Collection '{collection_name}' created with {vector_size}-dimension vectors")

    # Find all markdown files
    docs_dir = "../frontend/docs"
    md_files = find_markdown_files(docs_dir)

    # Process each file
    all_points = []
    point_id = 0

    for file_path in md_files:
        print(f"Processing: {file_path}")

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Strip frontmatter
        content_without_frontmatter = strip_frontmatter(content)

        # Get relative path for source tracking
        relative_path = str(file_path.relative_to(Path(docs_dir).parent))

        # Chunk the content
        chunks = chunk_text(content_without_frontmatter)
        print(f"  - Created {len(chunks)} chunks")

        # Create embeddings and points for each chunk
        for i, chunk in enumerate(chunks):
            # Get embedding
            try:
                embedding = get_embedding(chunk)
            except Exception as e:
                print(f"  - Error getting embedding for chunk {i}: {e}")
                continue

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": relative_path,
                    "page_url": f"/docs/{relative_path.replace('.md', '')}",
                    "chunk_index": i
                }
            )

            all_points.append(point)
            point_id += 1

    # Upload points to Qdrant
    if all_points:
        print(f"Uploading {len(all_points)} points to Qdrant...")

        # Upload in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(all_points), batch_size):
            batch = all_points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
            print(f"  Uploaded batch {i//batch_size + 1}/{(len(all_points)-1)//batch_size + 1}")

        print(f"Ingestion complete! Uploaded {len(all_points)} points to Qdrant collection '{collection_name}'")
    else:
        print("No points to upload - check if files were processed correctly")


if __name__ == "__main__":
    main()
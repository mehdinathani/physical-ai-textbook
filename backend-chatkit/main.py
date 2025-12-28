import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, AsyncIterator, List
from dataclasses import dataclass, field

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel

# Import from chatkit
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.store import Store
from chatkit.types import ThreadMetadata, ThreadItem, Page
from chatkit.agents import AgentContext, stream_agent_response, ThreadItemConverter

# Import for RAG (Qdrant + Google embeddings)
from qdrant_client import QdrantClient
import google.generativeai as genai

# Load .env from project root and backend-chatkit directory
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv(Path(__file__).parent / ".env")  # Also load from backend-chatkit/.env


@dataclass
class ThreadState:
    thread: ThreadMetadata
    items: list[ThreadItem] = field(default_factory=list)


class MemoryStore(Store[dict]):
    """Thread-safe in-memory store matching official ChatKit implementation"""

    def __init__(self) -> None:
        self._threads: dict[str, ThreadState] = {}
        self._attachments: dict[str, Any] = {}

    def generate_thread_id(self, context: dict) -> str:
        return f"thread_{uuid.uuid4().hex[:12]}"

    def generate_item_id(self, item_type: str, thread: ThreadMetadata, context: dict) -> str:
        new_id = f"{item_type}_{uuid.uuid4().hex[:12]}"
        print(f"[Store] generate_item_id: type={item_type}, id={new_id}")
        return new_id

    def _get_items(self, thread_id: str) -> list[ThreadItem]:
        state = self._threads.get(thread_id)
        return state.items if state else []

    async def load_thread(self, thread_id: str, context: dict) -> ThreadMetadata:
        state = self._threads.get(thread_id)
        if state:
            return state.thread.model_copy(deep=True)
        # Create new thread
        thread = ThreadMetadata(
            id=thread_id,
            created_at=datetime.now(timezone.utc),
            metadata={}
        )
        self._threads[thread_id] = ThreadState(thread=thread.model_copy(deep=True), items=[])
        return thread

    async def save_thread(self, thread: ThreadMetadata, context: dict) -> None:
        state = self._threads.get(thread.id)
        if state:
            state.thread = thread.model_copy(deep=True)
        else:
            self._threads[thread.id] = ThreadState(
                thread=thread.model_copy(deep=True),
                items=[]
            )

    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: dict,
    ) -> Page[ThreadItem]:
        items = [item.model_copy(deep=True) for item in self._get_items(thread_id)]

        # Sort by created_at
        items.sort(
            key=lambda i: getattr(i, "created_at", datetime.now(timezone.utc)),
            reverse=(order == "desc"),
        )

        # Handle pagination with 'after' cursor
        start = 0
        if after:
            index_map = {item.id: idx for idx, item in enumerate(items)}
            start = index_map.get(after, -1) + 1

        slice_items = items[start: start + limit + 1]
        has_more = len(slice_items) > limit

        result_items = slice_items[:limit]
        print(f"[Store] Returning {len(result_items)} items for thread {thread_id}, has_more={has_more}")

        return Page(
            data=result_items,
            has_more=has_more,
            after=slice_items[-1].id if has_more and slice_items else None
        )

    async def add_thread_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
        state = self._threads.get(thread_id)
        if not state:
            await self.load_thread(thread_id, context)
            state = self._threads[thread_id]

        # Debug: log item details
        item_type = type(item).__name__
        content_preview = ""
        if hasattr(item, 'content') and item.content:
            for part in item.content:
                if hasattr(part, 'text'):
                    content_preview = part.text[:50] + "..." if len(part.text) > 50 else part.text
                    break
        safe_print(f"[Store] add_thread_item: id={item.id}, type={item_type}, content='{content_preview}'")

        # Check if item exists, update if so
        for i, existing in enumerate(state.items):
            if existing.id == item.id:
                state.items[i] = item.model_copy(deep=True)
                print(f"[Store] Updated existing item {item.id}")
                return

        state.items.append(item.model_copy(deep=True))
        print(f"[Store] Added NEW item {item.id}, total items: {len(state.items)}")

    async def save_item(self, thread_id: str, item: ThreadItem, context: dict) -> None:
        await self.add_thread_item(thread_id, item, context)

    async def load_item(self, thread_id: str, item_id: str, context: dict) -> ThreadItem:
        for item in self._get_items(thread_id):
            if item.id == item_id:
                return item.model_copy(deep=True)
        raise ValueError(f"Item {item_id} not found")

    async def delete_thread_item(self, thread_id: str, item_id: str, context: dict) -> None:
        state = self._threads.get(thread_id)
        if state:
            state.items = [i for i in state.items if i.id != item_id]

    async def load_threads(self, limit: int, after: str | None, order: str, context: dict) -> Page[ThreadMetadata]:
        threads = [s.thread.model_copy(deep=True) for s in self._threads.values()]
        return Page(data=threads[-limit:], has_more=False)

    async def delete_thread(self, thread_id: str, context: dict) -> None:
        self._threads.pop(thread_id, None)

    async def save_attachment(self, attachment: Any, context: dict) -> None:
        self._attachments[attachment.id] = attachment

    async def load_attachment(self, attachment_id: str, context: dict) -> Any:
        if attachment_id not in self._attachments:
            raise ValueError(f"Attachment {attachment_id} not found")
        return self._attachments[attachment_id]

    async def delete_attachment(self, attachment_id: str, context: dict) -> None:
        self._attachments.pop(attachment_id, None)


# Helper function to safely print Unicode characters on Windows
def safe_print(message: str) -> None:
    """Print with Unicode support, handling Windows console encoding issues."""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: encode to ASCII with replacement for problematic characters
        print(message.encode('ascii', 'replace').decode('ascii'))


# RAG: Embedding generation function (same as main backend)
def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text using Google's text-embedding-004 model."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    genai.configure(api_key=google_api_key)

    result = genai.embed_content(
        model="models/text-embedding-004",
        content=[text],
        task_type="RETRIEVAL_QUERY"
    )

    return result['embedding'][0]


# RAG: Initialize Qdrant client
def get_qdrant_client() -> QdrantClient:
    """Initialize and return Qdrant client."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not all([qdrant_url, qdrant_api_key]):
        raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY")

    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=30
    )


# Gemini model via LiteLLM
# Note: gemini-2.0-flash has quota issues, using gemini-2.5-flash-lite which works
gemini_model = LitellmModel(
    model="gemini/gemini-2.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
)


class GeminiChatKitServer(ChatKitServer[dict]):
    def __init__(self, data_store: Store):
        super().__init__(data_store)

        self.assistant_agent = Agent[AgentContext](
            name="Gemini Assistant",
            instructions="You are an expert tutor for Physical AI & Humanoid Robotics. Help students understand concepts related to robotics, ROS2, Isaac Sim, and humanoid development. Be clear, educational, and encourage hands-on learning.",
            model=gemini_model,
        )
        self.converter = ThreadItemConverter()

    async def respond(self, thread: ThreadMetadata, input: Any, context: dict) -> AsyncIterator:
        from chatkit.types import (
            ThreadItemAddedEvent, ThreadItemDoneEvent,
            AssistantMessageItem
        )

        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )

        # Load all thread items and convert using ChatKit's converter
        page = await self.store.load_thread_items(thread.id, None, 100, "asc", context)
        all_items = list(page.data)

        # Add current input to the list if provided
        if input:
            all_items.append(input)

        print(f"[Server] Processing {len(all_items)} items for agent")

        # ============================================================
        # RAG: Extract user's latest message for context retrieval
        # ============================================================
        user_message = ""
        if input and hasattr(input, 'content') and input.content:
            for part in input.content:
                if hasattr(part, 'text'):
                    user_message = part.text
                    break

        rag_context = ""
        if user_message:
            try:
                print(f"[Server] RAG: Searching for context for query: {user_message[:50]}...")

                # Get embedding for user's message
                query_embedding = get_embedding(user_message)

                # Search Qdrant for relevant textbook content
                qdrant_client = get_qdrant_client()
                search_results = qdrant_client.query_points(
                    collection_name="textbook_docs",
                    query=query_embedding,
                    limit=3,  # Top 3 most relevant chunks
                    with_payload=True
                ).points

                if search_results:
                    print(f"[Server] RAG: Found {len(search_results)} relevant chunks")
                    context_parts = []
                    for result in search_results:
                        payload = result.payload
                        source = payload.get('source', 'Unknown')
                        text = payload.get('text', '')
                        context_parts.append(f"[From {source}]:\n{text}")

                    rag_context = "\n\n".join(context_parts)
                else:
                    print("[Server] RAG: No relevant context found")

            except Exception as e:
                print(f"[Server] RAG: Error during search: {e}")
                # Continue without RAG context if error occurs

        # Update agent instructions with RAG context if available
        if rag_context:
            self.assistant_agent.instructions = f"""You are an expert tutor for Physical AI & Humanoid Robotics.
Use the following context from the textbook to answer the user's question accurately.

IMPORTANT RESPONSE FORMAT:
- Start with "📚" indicator (without the brackets text, just the emoji)
- Keep responses SHORT and conversational (2-3 lines maximum)
- Be concise and clear, this is a chat not an essay
- Only expand if the user explicitly asks for more details

TEXTBOOK CONTEXT:
{rag_context}

Help students understand concepts clearly and encourage hands-on learning."""
            print("[Server] RAG: Updated agent with textbook context")
            print(f"[Server] RAG: Context preview: {rag_context[:200]}...")
        else:
            # Reset to original instructions if no context
            self.assistant_agent.instructions = """You are an expert tutor for Physical AI & Humanoid Robotics.

IMPORTANT RESPONSE FORMAT:
- Start with "💭" indicator (without the brackets text, just the emoji)
- Keep responses SHORT and conversational (2-3 lines maximum)
- Be concise and clear, this is a chat not an essay
- Only expand if the user explicitly asks for more details

Help students understand concepts related to robotics, ROS2, Isaac Sim, and humanoid development. Be clear, educational, and encourage hands-on learning."""
            print("[Server] RAG: No context found, using general knowledge")

        # Convert thread items to agent input format using ChatKit's converter
        agent_input = await self.converter.to_agent_input(all_items) if all_items else []

        print(f"[Server] Converted to {len(agent_input)} agent input items")

        result = Runner.run_streamed(
            self.assistant_agent,
            agent_input,
            context=agent_context,
        )

        # Track ID mappings to ensure unique IDs (LiteLLM/Gemini may reuse IDs)
        id_mapping: dict[str, str] = {}
        event_count = 0
        seen_assistant_content = False
        accumulated_text = ''  # Track accumulated delta text for final completion

        try:
            async for event in stream_agent_response(agent_context, result):
                event_count += 1
                # Debug: log each event type
                print(f"[Server] Yielding event #{event_count}: {event.type}")
                if hasattr(event, 'item'):
                    print(f"[Server]   item.id: {event.item.id if hasattr(event.item, 'id') else 'N/A'}")
                if hasattr(event, 'update'):
                    safe_print(f"[Server]   update: {event.update}")

                # Track if we've seen any assistant content
                if event.type == "thread.item.added":
                    if isinstance(event.item, AssistantMessageItem):
                        seen_assistant_content = True
                        safe_print(f"[Server] Assistant item added - content preview: {str(event.item.content)[:100] if event.item.content else 'empty'}")

                # Fix potential ID collisions from LiteLLM/Gemini
                if event.type == "thread.item.added":
                    if isinstance(event.item, AssistantMessageItem):
                        old_id = event.item.id
                        # Generate unique ID if we haven't seen this response ID before
                        if old_id not in id_mapping:
                            new_id = self.store.generate_item_id("message", thread, context)
                            id_mapping[old_id] = new_id
                            print(f"[Server] Mapping ID {old_id} -> {new_id}")
                        event.item.id = id_mapping[old_id]
                elif event.type == "thread.item.done":
                    if isinstance(event.item, AssistantMessageItem):
                        old_id = event.item.id
                        if old_id in id_mapping:
                            event.item.id = id_mapping[old_id]
                        # WORKAROUND: Update content with accumulated text from deltas
                        if accumulated_text and event.item.content and not any(
                            c.text for c in event.item.content if c.type in ('text', 'output_text')
                        ):
                            # Add accumulated deltas as text content
                            from chatkit.types import OutputText
                            event.item.content.append(OutputText(text=accumulated_text))
                            print(f"[Server] ADDED accumulated text to thread.item.done")
                elif event.type == "thread.item.updated":
                    if event.item_id in id_mapping:
                        event.item_id = id_mapping[event.item_id]
                    # Accumulate text from deltas for final completion
                    if hasattr(event, 'update') and hasattr(event.update, 'delta'):
                        if event.update.type == 'assistant_message.content_part.text_delta' and hasattr(event.update.delta, 'text'):
                            accumulated_text += event.update.delta.text
                            safe_print(f"[Server] Accumulated text so far: {accumulated_text[:50]}...")

                yield event
        except Exception as e:
            print(f"[Server] ERROR in stream_agent_response: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# Initialize FastAPI app
app = FastAPI(title="ChatKit Gemini")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://physical-ai-textbook-git-master-mehdinathanis-projects.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChatKit server
store = MemoryStore()
server = GeminiChatKitServer(store)


def convert_camelcase_to_snakecase(data):
    """Convert camelCase keys to snake_case for ChatKit v1.3.0 compatibility"""
    import json
    import re

    def camel_to_snake(name):
        # Convert camelCase to snake_case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Convert the key from camelCase to snake_case
            new_key = camel_to_snake(key) if not key.startswith('_') else key
            # Recursively convert nested structures
            new_dict[new_key] = convert_camelcase_to_snakecase(value)

        # Special handling for 'type' field to convert action names
        if 'type' in new_dict:
            type_val = new_dict['type']
            if '.' in type_val:
                prefix, action = type_val.split('.', 1)
                new_dict['type'] = f"{prefix}.{camel_to_snake(action)}"

        return new_dict
    elif isinstance(data, list):
        return [convert_camelcase_to_snakecase(item) for item in data]
    else:
        return data


@app.post("/chatkit")
async def chatkit_endpoint(request: Request):
    import traceback as tb
    body = await request.body()
    print(f"\n{'='*80}")
    print(f"[CHATKIT ENDPOINT] Received request")
    print(f"{'='*80}")
    print(f"[CHATKIT ENDPOINT] Body length: {len(body)} bytes")
    print(f"[CHATKIT ENDPOINT] Request body preview:", body[:500] if len(body) > 500 else body)

    try:
        import json
        body_json = json.loads(body)
        print(f"[CHATKIT ENDPOINT] Original body: {json.dumps(body_json, indent=2)}")

        # Convert camelCase to snake_case for backend compatibility
        converted_json = convert_camelcase_to_snakecase(body_json)
        print(f"[CHATKIT ENDPOINT] Converted body: {json.dumps(converted_json, indent=2)}")

        # Re-encode to bytes
        body = json.dumps(converted_json).encode('utf-8')
    except Exception as e:
        print(f"[CHATKIT ENDPOINT] Failed to parse/convert body: {e}")
        import traceback
        traceback.print_exc()

    print(f"{'='*80}\n")

    try:
        result = await server.process(body, {})
        print(f"[CHATKIT ENDPOINT] SUCCESS - Process succeeded")
        if isinstance(result, StreamingResult):
            return StreamingResponse(result, media_type="text/event-stream")
        return Response(content=result.json, media_type="application/json")
    except Exception as e:
        print(f"[CHATKIT ENDPOINT] ERROR: {type(e).__name__}: {str(e)}")
        print(f"[CHATKIT ENDPOINT] Error details:", type(e).__name__)
        import traceback
        traceback.print_exc()
        raise


@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-2.5-flash-lite"}


@app.get("/debug/threads")
async def debug_threads():
    """Debug endpoint to view all stored items"""
    result = {}
    for thread_id, state in store._threads.items():
        items = []
        for item in state.items:
            item_data = {
                "id": item.id,
                "type": type(item).__name__,
                "created_at": str(getattr(item, 'created_at', 'N/A')),
            }
            # Extract content
            if hasattr(item, 'content') and item.content:
                content_parts = []
                for part in item.content:
                    if hasattr(part, 'text'):
                        content_parts.append(part.text)
                item_data["content"] = content_parts
            items.append(item_data)
        result[thread_id] = {
            "thread": {"id": state.thread.id, "created_at": str(state.thread.created_at)},
            "items": items,
            "item_count": len(items)
        }
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    print(f"Starting ChatKit Gemini server at http://localhost:{port}")
    print(f"Health check: http://localhost:{port}/health")
    print(f"ChatKit endpoint: http://localhost:{port}/chatkit")
    uvicorn.run(app, host="0.0.0.0", port=port)

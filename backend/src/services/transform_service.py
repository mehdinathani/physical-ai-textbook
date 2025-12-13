import google.generativeai as genai
import os
from typing import Optional
from pydantic import BaseModel

# Configure the API key from environment
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class ContentTransformationRequest(BaseModel):
    source_content: str
    transformation_type: str  # "urdu-translation", "hardware-personalization", "software-personalization"
    preserve_formatting: bool = True

class ContentTransformationResponse(BaseModel):
    transformed_content: str
    transformation_type: str
    processing_time_ms: int

class TransformService:
    def __init__(self):
        # Initialize the Gemini model
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    async def transform_content(self, request: ContentTransformationRequest) -> ContentTransformationResponse:
        """
        Transform content based on the specified transformation type while preserving formatting
        """
        import time
        start_time = time.time()

        # Construct the appropriate prompt based on transformation type
        prompt = self._build_prompt(request.source_content, request.transformation_type, request.preserve_formatting)

        try:
            # Generate content using Gemini
            response = await self.model.generate_content_async(prompt)

            processing_time = int((time.time() - start_time) * 1000)

            return ContentTransformationResponse(
                transformed_content=response.text if response.text else request.source_content,
                transformation_type=request.transformation_type,
                processing_time_ms=processing_time
            )
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            # In case of error, return original content
            return ContentTransformationResponse(
                transformed_content=request.source_content,
                transformation_type=request.transformation_type,
                processing_time_ms=processing_time
            )

    def _build_prompt(self, content: str, transformation_type: str, preserve_formatting: bool) -> str:
        """
        Build the appropriate prompt based on transformation type
        """
        if transformation_type == "urdu-translation":
            instruction = "Translate this technical content to technical Urdu while preserving all HTML tags, structure, and formatting exactly as they are. Only translate the text content between HTML tags. Use appropriate technical terminology in Urdu."
        elif transformation_type == "hardware-personalization":
            instruction = "Rephrase this content to emphasize hardware/circuit implementation aspects, component specifications, and physical implementation details while preserving all HTML tags, structure, and formatting exactly as they are."
        elif transformation_type == "software-personalization":
            instruction = "Rephrase this content to emphasize software/code implementation aspects, algorithms, and programming details while preserving all HTML tags, structure, and formatting exactly as they are."
        else:
            # Default to no transformation if type is unknown
            instruction = "Return the content as is, preserving all HTML tags, structure, and formatting."

        if preserve_formatting:
            return f"IMPORTANT: Preserve all HTML tags, formatting, code blocks, image references, links, and structural elements exactly as they are. Only modify the text content between HTML tags.\n\n{instruction}\n\nContent to transform:\n{content}"
        else:
            return f"{instruction}\n\nContent to transform:\n{content}"
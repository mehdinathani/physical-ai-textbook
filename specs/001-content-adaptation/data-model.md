# Data Model: Intelligent Content Adaptation System

**Feature**: Content Adaptation System
**Date**: 2025-12-13

## Key Entities

### ContentTransformationRequest

Represents a request to transform content with specific parameters.

**Fields**:
- `id` (string): Unique identifier for the transformation request
- `sourceContent` (string): Original markdown content to be transformed
- `transformationType` (enum): Type of transformation ("urdu-translation", "hardware-personalization", "software-personalization")
- `preserveFormatting` (boolean): Whether to preserve markdown formatting, code blocks, and images (default: true)
- `createdAt` (timestamp): When the request was created

**Validation Rules**:
- `sourceContent` must not be empty
- `transformationType` must be one of the allowed values
- `sourceContent` length must be within API limits

### ContentTransformationResponse

Represents the result of a content transformation.

**Fields**:
- `id` (string): Matches the request ID
- `transformedContent` (string): The transformed content with preserved formatting
- `transformationType` (enum): Type of transformation applied
- `processingTimeMs` (number): Time taken to process the request in milliseconds
- `createdAt` (timestamp): When the response was generated

**Validation Rules**:
- `transformedContent` must not be empty
- `processingTimeMs` must be a positive number
- Content structure must match the original (same number of code blocks, images, etc.)

### PersonalizationPreference

Represents the user's current content view preference during their session.

**Fields**:
- `userId` (string): Session-based identifier (not persistent)
- `viewType` (enum): Current view preference ("default", "hardware-engineer", "software-engineer")
- `updatedAt` (timestamp): When the preference was last updated

**Validation Rules**:
- `viewType` must be one of the allowed values
- `userId` follows session-based format

## State Transitions

### Content Transformation Flow
1. `ContentTransformationRequest` is created when user initiates translation/personalization
2. Request is sent to backend API
3. Backend processes request using Google Gemini
4. `ContentTransformationResponse` is generated
5. Frontend updates content display with transformed content

### Personalization Preference Flow
1. User selects view preference (Hardware/Software Engineer)
2. `PersonalizationPreference` is updated in local state
3. Next transformation request includes the preference
4. Content is transformed according to the selected view type

## API Contracts

### Transformation Request Format
```json
{
  "sourceContent": "string (markdown content)",
  "transformationType": "urdu-translation|hardware-personalization|software-personalization",
  "preserveFormatting": true
}
```

### Transformation Response Format
```json
{
  "transformedContent": "string (transformed markdown content)",
  "transformationType": "urdu-translation|hardware-personalization|software-personalization",
  "processingTimeMs": 1234
}
```
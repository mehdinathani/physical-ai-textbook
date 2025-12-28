---
sidebar_position: 1
---

# Voice to Action with Whisper

Voice interfaces represent a paradigm shift in human-robot interaction, enabling communication through natural speech rather than complex control interfaces. OpenAI's Whisper speech recognition system has revolutionized this space by providing highly accurate, multilingual transcription that can be deployed on edge devices or in the cloud. This chapter covers the complete voice-to-action pipeline, from audio capture through speech recognition to robot command execution.

## Foundations of Speech Recognition

Automatic Speech Recognition (ASR) has undergone remarkable transformation over the past decade, evolving from systems requiring carefully curated acoustic models to end-to-end neural architectures that learn directly from audio data. Understanding these foundations is essential for effectively integrating speech recognition into robotic systems.

### The Speech Recognition Pipeline

The speech recognition pipeline consists of several interconnected stages that transform raw audio into actionable robot commands. At the input stage, audio is captured through microphones and converted from acoustic pressure waves into digital samples. This conversion process, governed by the Nyquist theorem, requires sampling rates at least twice the highest frequency component of interest. For speech recognition, a 16kHz sampling rate captures the full range of human speech frequencies, though higher rates may be used for improved audio quality.

The preprocessing stage normalizes audio characteristics and prepares the signal for neural network processing. This includes amplitude normalization to handle variations in speaker distance and microphone sensitivity, band-pass filtering to remove irrelevant frequency components, and voice activity detection to identify segments containing speech versus silence or background noise. The choice of preprocessing parameters significantly impacts recognition accuracy, particularly in challenging acoustic environments typical of real-world robotic applications.

Feature extraction converts the preprocessed audio into representations suitable for neural network processing. Traditional approaches used mel-frequency cepstral coefficients (MFCCs) that approximate human auditory perception, while modern end-to-end systems often use mel-spectrograms or raw waveform inputs. The Whisper architecture uses a mel-spectrogram computed from 25-millisecond windows with 10-millisecond overlaps, capturing both spectral and temporal characteristics of speech.

The neural network model processes these features to produce text output. Whisper uses an encoder-decoder transformer architecture where the encoder processes mel-spectrogram features through multiple transformer layers, and the decoder generates autoregressive text predictions. This architecture provides strong contextual understanding, enabling accurate recognition of words in context rather than in isolation.

### Whisper Architecture and Capabilities

Whisper represents a significant advance in speech recognition technology, trained on approximately 680,000 hours of multilingual and multitask supervised data collected from the web. This massive training corpus enables several unique capabilities that distinguish Whisper from previous speech recognition systems.

The encoder component processes 80-channel mel-spectrograms through a series of convolutional layers followed by transformer encoder layers. The convolutional layers perform initial feature extraction and downsampling, reducing the temporal resolution while increasing feature dimensionality. The transformer layers then process these features using self-attention mechanisms that capture long-range dependencies in the audio signal.

The decoder operates as an autoregressive language model, generating text tokens one at a time based on the encoder's output and previously generated tokens. During training, the model learns to predict the next token given the audio encoder output and preceding tokens. This training approach enables the model to generate coherent text that considers both the acoustic evidence and linguistic context.

Whisper offers multiple model sizes ranging from tiny (39 MB) to large (1550 MB), trading off between speed, memory requirements, and recognition accuracy. For robotic applications, the small or medium models often provide the best balance, offering near-large-model accuracy with significantly lower computational requirements. The following table summarizes the available models and their characteristics:

| Model | Parameters | Memory | Relative Speed | Best For |
|-------|------------|--------|----------------|----------|
| Tiny | 39M | ~1GB RAM | 32x | Simple commands, high-latency tolerance |
| Base | 74M | ~1GB RAM | 16x | Standard command recognition |
| Small | 244M | ~2GB RAM | 6x | Balanced performance |
| Medium | 769M | ~4GB RAM | 2x | High accuracy needs |
| Large | 1550M | ~8GB RAM | 1x | Maximum accuracy |

### Audio Capture and Preprocessing

Effective speech recognition begins with high-quality audio capture. The choice of microphone, its placement, and the acoustic environment all significantly impact recognition accuracy. For humanoid robots, microphone placement requires careful consideration of head design, body occlusion, and operational environments.

```python
import numpy as np
import sounddevice as sd
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Callable
import threading
import queue

@dataclass
class AudioConfig:
    """Configuration for audio capture and processing."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 2048  # Samples per buffer
    dtype: str = 'float32'
    buffer_duration: float = 0.5  # Seconds of audio per buffer
    vad_threshold: float = 0.02  # Voice activity threshold
    pre_emphasis: float = 0.97

class AudioCapture:
    """
    Captures and preprocesses audio for speech recognition.
    """

    def __init__(self, config: AudioConfig, on_audio_ready: Callable):
        self.config = config
        self.on_audio_ready = on_audio_ready
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None

    def start(self):
        """Start audio capture."""
        self.is_recording = True

        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.chunk_size,
            callback=self._audio_callback
        )

        self.stream.start()
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.start()

    def stop(self):
        """Stop audio capture."""
        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio stream status: {status}")

        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def _process_audio(self):
        """Process audio frames for speech recognition."""
        while self.is_recording or not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                self._process_frame(audio_data)
            except queue.Empty:
                continue

    def _process_frame(self, frame):
        """Process a single audio frame."""
        # Apply preprocessing
        processed = self._preprocess(frame)

        # Voice activity detection
        if self._is_voice_activity(processed):
            self.on_audio_ready(processed)

    def _preprocess(self, audio):
        """Apply audio preprocessing."""
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Apply pre-emphasis filter
        if self.config.pre_emphasis > 0:
            audio = np.append(audio[0], audio[1:] - self.config.pre_emphasis * audio[:-1])

        # Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio

    def _is_voice_activity(self, audio):
        """Detect voice activity based on energy."""
        energy = np.mean(audio**2)
        return energy > self.config.vad_threshold

class MelSpectrogramExtractor:
    """
    Extracts mel-spectrogram features for Whisper.
    """

    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Precompute mel filterbank
        self.mel_filter = self._create_mel_filter()

    def _create_mel_filter(self):
        """Create mel filterbank matrix."""
        # Compute mel frequencies
        mel_low = 0
        mel_high = 2595 * np.log10(1 + self.sample_rate / 2 / 700)
        mel_points = np.linspace(mel_low, mel_high, self.n_mels + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)

        # Convert to FFT bin indices
        bin_edges = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        # Create filterbank
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            for j in range(bin_edges[i], bin_edges[i + 1]):
                filterbank[i, j] = (
                    (j - bin_edges[i]) /
                    (bin_edges[i + 1] - bin_edges[i])
                )
            for j in range(bin_edges[i + 1], bin_edges[i + 2]):
                filterbank[i, j] = (
                    (bin_edges[i + 2] - j) /
                    (bin_edges[i + 2] - bin_edges[i + 1])
                )

        return filterbank

    def extract(self, audio):
        """
        Extract mel-spectrogram from audio.

        Args:
            audio: 1D numpy array of audio samples

        Returns:
            Mel-spectrogram as numpy array
        """
        # Compute STFT
        window = signal.windows.hann(self.n_fft)
        frequencies, times, stft = signal.stft(
            audio,
            fs=self.sample_rate,
            window=window,
            nfft=self.n_fft,
            noverlap=self.n_fft - self.hop_length
        )

        # Compute power spectrogram
        spectrogram = np.abs(stft) ** 2

        # Apply mel filterbank
        mel_spec = self.mel_filter @ spectrogram

        # Log-scale
        mel_spec = np.log10(mel_spec + 1e-10)

        return mel_spec
```

## Integrating Whisper with Robotic Systems

Integrating Whisper into a robotic system requires careful consideration of deployment location, latency requirements, and error handling. The choice between edge deployment, cloud processing, or hybrid approaches depends on the robot's computational resources, network connectivity, and application requirements.

### Local Deployment with ONNX Runtime

For applications requiring low latency or operating in environments with limited connectivity, local deployment of Whisper provides the most reliable performance. The ONNX (Open Neural Network Exchange) format enables efficient inference on various hardware platforms, from CPUs to specialized AI accelerators.

```python
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import time

class LocalWhisperRecognizer:
    """
    Local Whisper speech recognition for robotics.
    """

    def __init__(self, model_name: str = "small", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.lock = threading.Lock()

        # Recognition settings
        self.max_new_tokens = 256
        self.do_sample = False
        self.temperature = 0.0  # Deterministic output

    def load_model(self):
        """Load Whisper model for local inference."""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        print(f"Loading Whisper {self.model_name} on {self.device}...")

        self.processor = WhisperProcessor.from_pretrained(
            f"openai/whisper-{self.model_name}"
        )

        self.model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{self.model_name}"
        )

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

        print(f"Whisper {self.model_name} loaded successfully")

    def transcribe(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: 1D numpy array of audio samples

        Returns:
            Dictionary containing transcription results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        with self.lock:
            start_time = time.time()

            # Process audio
            input_features = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    no_repeat_ngram_size=3
                )

            # Decode output
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            inference_time = time.time() - start_time

            return {
                "text": transcription,
                "inference_time": inference_time,
                "model": self.model_name
            }

    def transcribe_streaming(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Process streaming audio for incremental recognition.

        Returns:
            Transcribed text if end of utterance detected, None otherwise
        """
        # Implementation would handle VAD and partial results
        pass

class WhisperConfig:
    """Configuration for Whisper integration."""

    def __init__(self):
        self.model_size = "small"
        self.device = "cpu"
        self.compute_type = "float32"  # or "int8" for faster inference
        self.beam_size = 5
        self.best_of = 5
        self.patience = 1
        self.length_penalty = -0.05
        self.no_repeat_ngram_size = 3
        self.temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.compression_ratio_threshold = 2.4
        self.log_prob_threshold = -1.0
        self.no_speech_threshold = 0.6
```

### Cloud-Based Transcription

For robots with reliable network connectivity, cloud-based transcription offers several advantages including access to the largest Whisper models, automatic updates, and reduced local computational requirements. The OpenAI API provides a simple interface for transcription services.

```python
import openai
import numpy as np
from typing import Dict, Any, Optional
import asyncio
import io
import wave

class CloudWhisperRecognizer:
    """
    Cloud-based Whisper transcription for robotics.
    """

    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    async def transcribe(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI API.

        Args:
            audio: 1D numpy array of audio samples

        Returns:
            Transcription result
        """
        import time
        start_time = time.time()

        # Convert numpy array to WAV bytes
        wav_bytes = self._audio_to_wav(audio, sample_rate=16000)

        # Upload to OpenAI
        response = await self.client.audio.transcriptions.async_create(
            model=self.model,
            file=("audio.wav", wav_bytes, "audio/wav"),
            language="en",
            response_format="verbose_json"
        )

        inference_time = time.time() - start_time

        return {
            "text": response.text,
            "duration": response.duration,
            "language": response.language,
            "inference_time": inference_time,
            "segments": [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text
                }
                for seg in response.segments
            ]
        }

    def _audio_to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        buffer = io.BytesIO()

        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
            wav_file.setframerate(sample_rate)

            # Convert float audio to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()
```

### Command Parsing and Intent Recognition

Transcribed text must be parsed into structured commands that the robot can execute. This involves intent recognition, entity extraction, and command validation.

```python
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import json

class RobotIntent(Enum):
    """Possible robot command intents."""
    NAVIGATE = "navigate"
    GRASP = "grasp"
    PLACE = "place"
    PICK_AND_PLACE = "pick_and_place"
    LOOK_AT = "look_at"
    FOLLOW = "follow"
    STOP = "stop"
    WAIT = "wait"
    FIND = "find"
    DESCRIBE = "describe"
    UNKNOWN = "unknown"

@dataclass
class ParsedCommand:
    """Structured representation of a voice command."""
    intent: RobotIntent
    entities: Dict[str, Any]
    confidence: float
    raw_text: str

class VoiceCommandParser:
    """
    Parses voice commands into structured robot commands.
    """

    def __init__(self):
        # Define command patterns with entity extraction
        self.patterns = {
            RobotIntent.NAVIGATE: [
                r"go to (?P<location>.+)",
                r"navigate to (?P<location>.+)",
                r"move to (?P<location>.+)",
                r"walk to (?P<location>.+)",
            ],
            RobotIntent.GRASP: [
                r"pick up (?P<object>.+)",
                r"grab (?P<object>.+)",
                r"grasp (?P<object>.+)",
                r"pick (?P<object>.+)",
            ],
            RobotIntent.PLACE: [
                r"put (?P<object>.+) (?P<location>.+)",
                r"place (?P<object>.+) (?P<location>.+)",
                r"set (?P<object>.+) (?P<location>.+)",
            ],
            RobotIntent.LOOK_AT: [
                r"look at (?P<target>.+)",
                r"look towards (?P<target>.+)",
                r"face (?P<target>.+)",
            ],
            RobotIntent.FOLLOW: [
                r"follow (?P<target>.+)",
                r"come with me",
                r"stay with (?P<target>.+)",
            ],
            RobotIntent.STOP: [
                r"stop",
                r"halt",
                r"freeze",
                r"pause",
            ],
            RobotIntent.WAIT: [
                r"wait",
                r"hold on",
                r"stay",
            ],
        }

    def parse(self, text: str) -> ParsedCommand:
        """
        Parse voice command text into structured command.

        Args:
            text: Raw transcription text

        Returns:
            ParsedCommand with intent and entities
        """
        text = text.lower().strip()

        # Try each intent pattern
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entities = match.groupdict()
                    return ParsedCommand(
                        intent=intent,
                        entities=entities,
                        confidence=self._compute_confidence(text, match),
                        raw_text=text
                    )

        # Unknown intent
        return ParsedCommand(
            intent=RobotIntent.UNKNOWN,
            entities={},
            confidence=0.0,
            raw_text=text
        )

    def _compute_confidence(self, text: str, match) -> float:
        """Compute confidence based on pattern match quality."""
        matched_length = match.end() - match.start()
        text_length = len(text)
        return matched_length / text_length if text_length > 0 else 0.0

    def entities_to_action(self, command: ParsedCommand) -> Dict[str, Any]:
        """
        Convert parsed command to robot action specification.
        """
        action = {
            "intent": command.intent.value,
            "confidence": command.confidence,
            "parameters": {}
        }

        if command.intent == RobotIntent.NAVIGATE:
            action["parameters"]["destination"] = command.entities.get("location", "")

        elif command.intent == RobotIntent.GRASP:
            action["parameters"]["target_object"] = command.entities.get("object", "")

        elif command.intent == RobotIntent.PLACE:
            action["parameters"]["object"] = command.entities.get("object", "")
            action["parameters"]["destination"] = command.entities.get("location", "")

        elif command.intent == RobotIntent.LOOK_AT:
            action["parameters"]["target"] = command.entities.get("target", "")

        elif command.intent == RobotIntent.FOLLOW:
            action["parameters"]["target"] = command.entities.get("target", "person")

        return action
```

## Complete Voice-to-Action Pipeline

The complete voice-to-action pipeline integrates audio capture, speech recognition, command parsing, and robot execution into a cohesive system.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, AudioFrame
from ai_msgs.msg import RobotCommand, CommandIntent
import json

class VoiceToActionNode(Node):
    """
    ROS 2 node for voice-to-action pipeline.
    """

    def __init__(self):
        super().__init__('voice_to_action')

        # Configuration
        self.declare_parameter('whisper_model', 'small')
        self.declare_parameter('use_cloud', False)
        self.declare_parameter('api_key', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('audio_device', '')

        self.model_name = self.get_parameter('whisper_model').value
        self.use_cloud = self.get_parameter('use_cloud').value
        self.api_key = self.get_parameter('api_key').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value

        # Initialize components
        self.audio_capture = None
        self.recognizer = None
        self.command_parser = VoiceCommandParser()

        # State
        self.is_listening = False
        self.last_transcription = ""

        # ROS interfaces
        self.command_pub = self.create_publisher(
            RobotCommand, '/robot/commands', 10
        )

        self.transcription_pub = self.create_publisher(
            String, '/voice/transcription', 10
        )

        self.status_pub = self.create_publisher(
            String, '/voice/status', 10
        )

        self.listen_sub = self.create_subscription(
            Bool, '/voice/listen', self.on_listen_command, 10
        )

        # Initialize recognizer
        self.initialize_recognizer()

        self.get_logger().info("Voice-to-Action node initialized")

    def initialize_recognizer(self):
        """Initialize speech recognizer based on configuration."""
        if self.use_cloud:
            self.recognizer = CloudWhisperRecognizer(self.api_key)
        else:
            self.recognizer = LocalWhisperRecognizer(self.model_name)
            self.recognizer.load_model()

    def on_listen_command(self, msg: Bool):
        """Handle listen command."""
        self.is_listening = msg.data

        if self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()

        status = json.dumps({
            "status": "listening" if self.is_listening else "idle"
        })
        self.status_pub.publish(String(data=status))

    def start_listening(self):
        """Start audio capture and recognition."""
        self.get_logger().info("Starting voice command listening")

        def on_audio_ready(audio):
            self.process_audio(audio)

        config = AudioConfig()
        self.audio_capture = AudioCapture(config, on_audio_ready)
        self.audio_capture.start()

    def stop_listening(self):
        """Stop audio capture."""
        if self.audio_capture:
            self.audio_capture.stop()
            self.audio_capture = None

    def process_audio(self, audio: np.ndarray):
        """Process audio and generate robot commands."""
        try:
            # Transcribe
            result = self.recognizer.transcribe(audio)
            text = result["text"].strip()

            if not text:
                return

            self.last_transcription = text

            # Publish transcription
            self.transcription_pub.publish(String(data=text))
            self.get_logger().info(f"Transcribed: {text}")

            # Parse command
            parsed = self.command_parser.parse(text)

            # Check confidence threshold
            if parsed.confidence < self.confidence_threshold:
                self.get_logger().warning(
                    f"Low confidence: {parsed.confidence:.2f}"
                )
                return

            # Convert to robot action
            action = self.command_parser.entities_to_action(parsed)

            # Create and publish command
            cmd = RobotCommand()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.intent = parsed.intent.value
            cmd.parameters = json.dumps(action["parameters"])
            cmd.confidence = parsed.confidence
            cmd.raw_command = text

            self.command_pub.publish(cmd)

        except Exception as e:
            self.get_logger().error(f"Error processing audio: {e}")
```

### Handling Recognition Errors and Clarification

Robust voice interfaces must handle recognition errors gracefully, including requesting clarification when confidence is low or when commands are ambiguous.

```python
class ClarificationHandler:
    """
    Handles clarification requests for ambiguous commands.
    """

    def __init__(self, tts_player):
        self.tts = tts_player
        self.pending_commands = {}
        self.max_clarifications = 2

    def handle_low_confidence(self, command: ParsedCommand, context: Dict) -> Optional[str]:
        """
        Generate clarification question for low confidence command.
        """
        clarification_questions = {
            RobotIntent.NAVIGATE: (
                "I'm not sure where you want me to go. "
                "Could you specify the location?"
            ),
            RobotIntent.GRASP: (
                "I didn't catch which object you want me to pick up. "
                "Please specify the object."
            ),
            RobotIntent.PLACE: (
                "I need clarification. What should I put where?"
            ),
            RobotIntent.UNKNOWN: (
                "I didn't understand that command. "
                "Could you please repeat it?"
            ),
        }

        question = clarification_questions.get(
            command.intent,
            "I'm not sure what you mean. Could you clarify?"
        )

        self.tts.speak(question)

        # Store pending command with clarification count
        command_id = hash(command.raw_text)
        self.pending_commands[command_id] = {
            "command": command,
            "clarifications": 0,
            "context": context
        }

        return question

    def process_clarification(self, command_id: str, clarification: str) -> Optional[ParsedCommand]:
        """
        Process user's clarification response.
        """
        if command_id not in self.pending_commands:
            return None

        pending = self.pending_commands[command_id]
        pending["clarifications"] += 1

        if pending["clarifications"] >= self.max_clarifications:
            del self.pending_commands[command_id]
            self.tts.speak(
                "I'm having trouble understanding. "
                "Please try again with a simpler command."
            )
            return None

        # Re-parse clarification
        clarified = self.command_parser.parse(clarification)

        # Check if clarification resolved the issue
        if clarified.confidence > 0.7:
            del self.pending_commands[command_id]
            self.tts.speak("Thank you for clarifying.")
            return clarified

        return None
```

## Key Takeaways

Voice interfaces enable intuitive human-robot interaction through natural language commands. The combination of Whisper for speech recognition and structured command parsing enables robust voice-controlled robotics.

- **Speech recognition foundations** include audio capture, preprocessing, and feature extraction
- **Whisper architecture** offers multiple model sizes balancing speed and accuracy
- **Local deployment** provides low latency and offline capability
- **Cloud processing** offers access to larger models with simpler deployment
- **Command parsing** extracts intent and entities from transcribed text
- **Error handling** requires clarification mechanisms for robust operation
- **ROS integration** enables modular voice control within larger robotic systems

With voice interfaces established, we can now explore how Large Language Models enable cognitive planning for complex robot tasks.

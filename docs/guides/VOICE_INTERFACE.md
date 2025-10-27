# 🎤 Morgan AI - Simplified Voice Interface

A streamlined voice assistant with a single-button interface for recording, processing, and playback.

## 🚀 Features

- **One-Button Operation**: Press and hold to record, release to process and play response
- **Automatic Pipeline**: STT → LLM → TTS processing happens automatically
- **Real-time Feedback**: Visual indicators for recording, processing, and playback states
- **Responsive Design**: Works on desktop and mobile devices
- **No Complex Setup**: No device selection or configuration required

## 📱 How to Use

1. **Access the Interface**: Navigate to `http://localhost:8000/voice`
2. **Record Your Message**: Press and hold the microphone button to start recording
3. **Stop Recording**: Release the button to stop recording and start processing
4. **Get Response**: The system will automatically:
   - Transcribe your speech to text
   - Generate an AI response
   - Convert the response to speech
   - Play it back automatically

## 🏗️ Architecture

### Frontend (core/static/voice_input.html)
- **Single Button Interface**: Clean, minimal design with one circular button
- **Real-time Status**: Visual feedback showing current state (idle, recording, processing)
- **Audio Playback**: Built-in audio player for TTS responses
- **Mobile Friendly**: Touch events and responsive design

### Backend (core/api/server.py)
- **Unified API**: `/api/audio` endpoint handles the complete pipeline
- **Service Orchestration**: Coordinates STT, LLM, and TTS services
- **Error Handling**: Robust error handling with user-friendly messages
- **Format Support**: Supports WebM, WAV, MP3, and other common audio formats

### Processing Pipeline
1. **Audio Recording** → WebM format with MediaRecorder API
2. **STT Processing** → Convert to text using Whisper models
3. **LLM Generation** → Generate AI response using external Ollama
4. **TTS Synthesis** → Convert response to speech using csm-streaming voices
5. **Audio Playback** → Play response audio in browser

## 🎯 API Endpoints

### GET /voice
Returns the simplified voice interface HTML page.

### POST /api/audio
Processes audio input through the complete pipeline.

**Parameters:**
- `file`: Audio file (WebM, WAV, MP3, etc.)
- `user_id`: User identifier (default: "voice_user")
- `device_type`: Device type (default: "microphone")
- `language`: Language code (default: "auto")

**Response:**
```json
{
  "transcription": "What you said",
  "ai_response": "AI generated response",
  "audio": "hex_encoded_audio_data",
  "metadata": {
    "user_id": "voice_user",
    "device_type": "microphone",
    "language": "auto",
    "processing_time": 2.5
  }
}
```

## 🔧 Configuration

The voice interface uses the following default settings:
- **STT Model**: Whisper large-v3 with VAD
- **LLM Model**: Llama 3.2 via external Ollama
- **TTS Voice**: "default" (American female)
- **Sample Rate**: 16kHz for recording, 22kHz for playback

## 🎨 Customization

### Changing the Voice
Modify the TTS voice in `core/api/server.py`:
```python
"voice": "default"  # Options: default, default, default, default
```

### Adjusting Language Detection
Change the default language in the frontend:
```javascript
language: 'auto'  // Options: auto, en, es, fr, de, etc.
```

## 🐛 Troubleshooting

### Common Issues

1. **Microphone Access Denied**
   - Ensure HTTPS is used in production
   - Check browser permissions
   - Try refreshing the page

2. **Audio Not Playing**
   - Check browser audio settings
   - Ensure audio context is allowed
   - Try different audio formats

3. **Processing Errors**
   - Check that all services are running and healthy
   - Verify network connectivity between services
   - Check service logs for detailed error messages

### Health Checks
- **Core Service**: http://localhost:8000/health
- **LLM Service**: http://localhost:8001/health
- **TTS Service**: http://localhost:8002/health
- **STT Service**: http://localhost:8003/health

## 📊 Performance

- **Recording**: Real-time with minimal latency
- **Processing**: 2-5 seconds depending on audio length
- **Audio Quality**: High-quality TTS with natural voices
- **Compatibility**: Works with all modern browsers

## 🔄 Service Dependencies

The voice interface requires all Morgan services to be running:
- ✅ **Core Service** (port 8000): API orchestration
- ✅ **LLM Service** (port 8001): AI response generation
- ✅ **TTS Service** (port 8002): Text-to-speech synthesis
- ✅ **STT Service** (port 8003): Speech-to-text transcription

## 🚀 Getting Started

1. **Start All Services**:
   ```bash
   docker-compose up -d
   ```

2. **Access Voice Interface**:
   ```
   http://localhost:8000/voice
   ```

3. **Test with Sample Audio**:
   - Press and hold the microphone button
   - Speak a short message
   - Release to see automatic processing and response

## 📝 Technical Notes

- **Audio Format**: WebM with Opus codec for recording
- **Processing**: Hex-encoded audio data for service communication
- **Error Handling**: Graceful degradation with user-friendly messages
- **State Management**: Client-side state tracking for UI updates
- **Browser Support**: Chrome, Firefox, Safari, Edge (modern versions)

---

🎯 **The result is a clean, one-button voice assistant that handles the complete conversation flow automatically!**

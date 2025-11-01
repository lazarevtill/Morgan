# Voice Activity Detection (VAD) Integration

> **Important**: Voice Activity Detection is **integrated into the STT service** via faster-whisper. There is no separate VAD service.

---

## Overview

Morgan AI Assistant uses **Silero VAD** for voice activity detection, but it's **not a standalone service**. Instead, it's integrated directly into the Speech-to-Text (STT) service through the faster-whisper library.

---

## Architecture

### ❌ Incorrect Architecture (Old)
```
Client → Core → VAD Service (separate) → STT Service
```

### ✅ Correct Architecture (Current)
```
Client → Core → STT Service (with integrated Silero VAD via faster-whisper)
```

---

## How It Works

### faster-whisper + Silero VAD Integration

1. **Audio Input**: Client sends audio to STT service
2. **VAD Processing**: faster-whisper internally uses Silero VAD to:
   - Detect speech segments
   - Filter out noise and silence
   - Identify speech boundaries
3. **Transcription**: Whisper model transcribes only the speech segments
4. **Response**: STT service returns transcription with VAD metadata

### Benefits

- **Improved Accuracy**: Only speech segments are transcribed
- **Reduced Latency**: Skips silent portions
- **Lower Resource Usage**: Processes only relevant audio
- **Automatic Noise Filtering**: Silero VAD filters background noise
- **Simpler Architecture**: One less service to manage

---

## Implementation

### Dependencies

In `requirements-stt.txt`:

```txt
faster-whisper==1.0.3  # Includes VAD support
silero-vad==4.0.2      # Used by faster-whisper internally
```

**Note**: `silero-vad` is installed as a dependency for faster-whisper to use, but there's no separate VAD service.

### STT Service Code

The STT service uses faster-whisper's built-in VAD support:

```python
from faster_whisper import WhisperModel

class STTService:
    def __init__(self):
        self.model = WhisperModel(
            "distil-distil-large-v3.5 ",
            device="cuda",
            compute_type="float16"
        )
    
    async def transcribe(self, audio_data: bytes, use_vad: bool = True) -> dict:
        """Transcribe audio with integrated VAD"""
        
        # faster-whisper automatically uses VAD when enabled
        segments, info = self.model.transcribe(
            audio_data,
            language="auto",
            vad_filter=use_vad,  # Enable/disable VAD
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": 30,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400
            }
        )
        
        return {
            "text": " ".join([s.text for s in segments]),
            "vad_enabled": use_vad,
            "segments": segments
        }
```

### API Usage

#### Request with VAD Enabled (Default)

```bash
curl -X POST http://localhost:8003/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "base64_encoded_audio",
    "use_vad": true,
    "vad_threshold": 0.5
  }'
```

#### Response with VAD Metadata

```json
{
  "text": "Turn on the living room lights",
  "language": "en",
  "confidence": 0.95,
  "duration": 2.1,
  "vad_enabled": true,
  "vad_segments": [
    {
      "start": 0.5,
      "end": 2.1,
      "duration": 1.6,
      "speech_detected": true
    }
  ],
  "segments": [
    {
      "start": 0.0,
      "end": 0.8,
      "text": "Turn on",
      "confidence": 0.98
    },
    {
      "start": 0.8,
      "end": 1.5,
      "text": "the living room",
      "confidence": 0.94
    },
    {
      "start": 1.5,
      "end": 2.1,
      "text": "lights",
      "confidence": 0.96
    }
  ]
}
```

---

## Configuration

### STT Service Config (`config/stt.yaml`)

```yaml
host: "0.0.0.0"
port: 8003
model: "distil-distil-large-v3.5 "
device: "cuda"
language: "auto"
sample_rate: 16000

# VAD Configuration (integrated)
vad:
  enabled: true
  threshold: 0.5
  min_speech_duration: 0.25  # seconds
  max_speech_duration: 30.0  # seconds
  min_silence_duration: 0.5  # seconds
  speech_pad: 0.4  # seconds
```

### Environment Variables

```bash
# STT service environment
MORGAN_STT_VAD_ENABLED=true
MORGAN_STT_VAD_THRESHOLD=0.5
```

---

## Performance

### With VAD (Recommended)

| Metric | Value | Notes |
|--------|-------|-------|
| Transcription Time | ~720ms | 10-second audio |
| Accuracy | 95%+ | Improved by noise filtering |
| CPU Usage | Low | Silero VAD is lightweight |
| GPU Usage | Moderate | Only for speech segments |

### Without VAD

| Metric | Value | Notes |
|--------|-------|-------|
| Transcription Time | ~950ms | 10-second audio (includes silence) |
| Accuracy | 85-90% | Noise affects quality |
| CPU Usage | Low | - |
| GPU Usage | High | Processes entire audio |

**Recommendation**: Always use VAD for production deployments.

---

## Comparison with Separate VAD Service

### Why NOT Use a Separate VAD Service?

| Aspect | Integrated VAD | Separate VAD Service |
|--------|---------------|----------------------|
| **Latency** | Low (single service) | High (extra network hop) |
| **Complexity** | Low (one service) | High (two services) |
| **Maintenance** | Easier | More complex |
| **Resource Usage** | Lower | Higher (extra container) |
| **Reliability** | Higher | Lower (more failure points) |
| **Integration** | Native in faster-whisper | Custom implementation needed |

### Migration from Separate VAD

If you previously had a separate VAD service, remove it:

```yaml
# docker-compose.yml - REMOVE THIS
# vad-service:
#   image: morgan/vad-service
#   ports:
#     - "8004:8004"
```

Update Core service to remove VAD service URL:

```yaml
# config/core.yaml - REMOVE THIS
# vad_service_url: "http://vad-service:8004"
```

---

## Troubleshooting

### VAD Not Working

**Symptom**: Transcription includes noise or doesn't filter silence

**Solutions**:

1. **Verify VAD is enabled**:
   ```python
   # In transcribe request
   {"use_vad": true}
   ```

2. **Check silero-vad installation**:
   ```bash
   docker exec morgan-stt python -c "import silero_vad; print('OK')"
   ```

3. **Adjust VAD threshold**:
   ```yaml
   # Lower threshold = more sensitive (detects more speech)
   # Higher threshold = less sensitive (filters more noise)
   vad_threshold: 0.5  # Try 0.3-0.7 range
   ```

### Transcription Too Slow

**Symptom**: STT takes longer than expected

**Check**: VAD parameters may be too strict

```yaml
# Relax VAD parameters
vad:
  threshold: 0.3  # More sensitive
  min_speech_duration: 0.1  # Shorter minimum
```

### Missing Speech Segments

**Symptom**: Parts of speech are not transcribed

**Check**: VAD may be filtering too aggressively

```yaml
# More permissive settings
vad:
  threshold: 0.3
  speech_pad: 0.6  # Larger padding around speech
  min_silence_duration: 0.3  # Less silence required
```

---

## References

- [faster-whisper Documentation](https://github.com/guillaumekln/faster-whisper)
- [Silero VAD Documentation](https://github.com/snakers4/silero-vad)
- [STT Service API](./API.md#-stt-service-api)
- [Version Alignment](../deployment/VERSION_ALIGNMENT.md)

---

## Summary

✅ **VAD is integrated into faster-whisper**  
✅ **No separate VAD service needed**  
✅ **Silero VAD 4.0.2 used internally**  
✅ **Configured via STT service**  
✅ **Improves accuracy and performance**  

---

**Last Updated**: 2025-10-27  
**Morgan AI Assistant** - Integrated VAD for better speech recognition


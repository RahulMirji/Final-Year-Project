
# Instructions: Intelligent Voice-Interactive Outfit Recommendation App

---

## ✅ Step-by-Step System Architecture

### 🎯 Overall System Objective

An Android app acting as a smart fashion assistant that:
- Continuously “mirrors” the user via camera.
- Automatically captures frames in background (no button clicks).
- Processes each image via a custom CNN to score the outfit and extract features.
- Feeds structured JSON (features + score) into Gemini API for conversational reasoning.
- Provides real-time, human-like voice feedback.
- Supports voice commands (e.g., “How do I look today?”).
- Includes a photo-based scoring mode with text feedback.

---

## ✅ 1️⃣ User Flow & Data Flow (High-Level)

```plaintext
[User opens app]
      ↓
[Home Screen]
  ├── Option 1: AI Assistant (Video Mode)
  │       ↓
  │   Continuous Camera Stream → Frame Sampler → CNN Inference → JSON Features + Score → Gemini API → LLM Text Response → TTS → Speaker
  │
  └── Option 2: Outfit Scorer (Photo Mode)
          ↓
     Capture Photo → CNN Inference → JSON Features + Score → Gemini API → LLM Text Explanation → Display Text

Additionally:
[User Voice Command]
  ↓
Microphone (Hotword-Triggered) → Voice-to-Text (Google Speech-to-Text / Whisper) → Gemini API → LLM Text Response → Text-to-Speech → Speaker
```

---

## ✅ 2️⃣ Core Components & Tools

| Component                       | Tool / Framework                                          |
| ------------------------------- | ---------------------------------------------------------- |
| Continuous Camera Stream        | Android CameraX API                                         |
| Frame Sampler (N-th frame)      | Custom frame buffer                                           |
| Clothing Detection + Scoring    | Custom CNN (MobileNet/EfficientNet trained locally)         |
| Feature Extraction              | CNN Outputs (shirt color, pant color, etc.)               |
| Structured JSON                 | Custom Python/JavaScript code                              |
| LLM Reasoning & Voice Feedback  | Gemini API                                                   |
| Text-to-Speech (TTS)            | Google Text-to-Speech API / Coqui TTS                     |
| Speech-to-Text (STT)            | Google Speech-to-Text API / Whisper                       |
| Local LLM fallback              | qwen2.5 or gemma3 via Ollama                              |
| Data Storage                    | SQLite (optional, for session history)                    |
| User Interface                  | React Native                                               |
| Background Processing           | Async Worker / Thread / Coroutine (Android)              |
| Model Inference                 | ONNX Runtime Mobile (for CNN model)                      |

---

## ✅ 3️⃣ Detailed Data Flow

### ▶️ AI Assistant (Video Mode)

1. 📸 CameraX API → Stream frames (30 fps).

2. ⚡ Every N-th frame → Send to CNN Model Inference (ONNX Runtime Mobile).

3. ✅ CNN Model → Output:

```json
{
  "shirt_color": "blue",
  "pant_color": "black",
  "shoes_type": "sneakers",
  "pose": "standing",
  "outfit_score": 85
}
```

4. 🌐 Send JSON → Gemini API (LLM Reasoning API).

5. Gemini API Response:

```json
{
  "response_text": "Hey Rahul, looking sharp today! I think your blue shirt complements the black pants well."
}
```

6. 🎙️ Text → TTS → Speaker Output.

7. (Optional) Log the session in local SQLite DB.

---

### ▶️ Photo Scorer Mode

1. 📸 User takes photo (CameraX).
2. ✅ CNN Model → Same inference → JSON Features + Score.
3. 🌐 Send JSON → Gemini API → Text response.
4. ✍️ Text Displayed on Screen + Optional Voice Output.

---

### ▶️ Voice Command Interaction

1. 🎙️ Hotword Detection (e.g., “Hey Assistant”).
2. 🎧 Google Speech-to-Text API / Whisper → Convert Voice to Text.
3. 🌐 Text Query → Gemini API → LLM Text Response.
4. 🎙️ Response → TTS → Speaker.

---

## ✅ 4️⃣ Why This Architecture Is Good

✔️ Low Latency →  
- CNN inference runs locally in milliseconds via ONNX Runtime Mobile.  
- Gemini API handles reasoning (fast).

✔️ Completely Hands-Free →  
- Hotword trigger + Continuous Mirror Mode makes it seamless.  
- No buttons needed → User-focused experience.

✔️ Modular →  
- Easy to update CNN or LLM independently.  
- Gemini API for reasoning enables powerful text generation.

✔️ Offline Fallback →  
- Use local LLM (qwen2.5:0.5b / gemma3:4b) if internet unavailable.

---

## ✅ 5️⃣ Example Pipeline Diagram

```plaintext
[CameraX API (Video Stream)]
      ↓
[Frame Sampler (Every 1 sec)]
      ↓
[ONNX CNN Model]
      ↓
JSON: {shirt_color, pant_color, shoes_type, pose, outfit_score}
      ↓
[Gemini API (LLM)]
      ↓
{response_text}
      ↓
[Text-to-Speech (TTS)]
      ↓
Speaker (Voice Output)
```

---

## ✅ 6️⃣ Next Steps

1. ✅ Train your CNN model locally (I can help you prepare the full training pipeline).
2. ✅ Export to ONNX.
3. ✅ Prototype CameraX stream + Frame Sampler on Android.
4. ✅ Integrate ONNX model inference → Get JSON output.
5. ✅ Set up Gemini API integration (easy REST API call).
6. ✅ Integrate Google Speech-to-Text & Text-to-Speech.
7. ✅ Build the React Native UI + Async background workers.

# ◢ SHOREN — AI Music Generator

Shoren is a cinematic AI music generation platform. It trains a neural network (LSTM/Transformer) on genre-specific musical data and generates expressive **MIDI files** you can download instantly.


### 🎵 Live Demo  
Experience SHOREN — AI Music Generator in action:  
👉 [https://codealpha-music-generation.onrender.com](https://codealpha-music-generation.onrender.com)



## Features
- 5 genres: Classical, Jazz, Ambient, Blues, Romantic
- Real neural network (TensorFlow LSTM) — trains live on each request
- Downloads real `.mid` MIDI files
- Cinematic Three.js frontend with live progress tracking
- No broken Models tab — clean, focused UI

## Quick Start

### Requirements
- Python 3.8+
- pip

### Install & Run

```bash
cd shoren
pip install -r requirements.txt
python app.py
```

Then open: **http://localhost:5000**

### Docker

```bash
docker build -t shoren .
docker run -p 5000:5000 shoren
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Start a generation job |
| GET | `/status/<job_id>` | Poll job status |
| GET | `/stream/<job_id>` | SSE progress stream |
| GET | `/download/<filename>` | Download MIDI file |
| GET | `/api/health` | Health check |
| GET | `/api/genres` | Genre info |
| GET | `/api/history` | Recent generations |
| GET | `/api/stats` | Server stats |

### Generate Request Body
```json
{
  "genre": "jazz",
  "num_notes": 64,
  "tempo": 120,
  "temperature": 0.8,
  "complexity": 3
}
```

## Generation Time
Each generation trains the model fresh (~30-120 seconds depending on complexity).
Complexity 3 (default) takes ~60-90s on CPU.

## Notes
- MIDI files saved to `static/audio/`
- All files named `shoren_<jobid>.mid`
- History kept in memory (50 most recent)

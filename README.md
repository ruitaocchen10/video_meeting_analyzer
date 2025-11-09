# Recruit Ready

An AI-powered interview preparation application that provides real-time feedback on posture, speech, and body language during mock interviews. The system combines computer vision, audio analysis, and an AI agent to simulate realistic interview scenarios with personalized feedback.

## 🎯 Project Overview

This project helps users practice and improve their interview skills by:
- **Real-time Monitoring**: Tracks posture, gestures, eye contact, and speech patterns
- **AI-Powered Feedback**: Provides instant, personalized feedback through an intelligent interviewer agent
- **Multiple Interviewer Personas**: Practice with different interviewer styles and difficulty levels
- **Comprehensive Analysis**: Analyzes both visual and audio metrics to give holistic feedback

## ✨ Features

### Computer Vision Analysis (`camera.py`)
- **Posture Detection**: Monitors shoulder angle, head tilt, and forward lean
- **Motion Tracking**: Tracks head and hand movement to detect fidgeting
- **Eye Contact Analysis**: Uses iris tracking to monitor eye contact with the camera
- **Real-time Visualization**: Displays metrics and annotations on video feed

### Speech Analysis (`voice.py`)
- **Speech Transcription**: Uses OpenAI Whisper for accurate speech-to-text
- **Speaking Rate**: Calculates words per minute (WPM) to detect pacing issues
- **Volume Analysis**: Monitors audio levels in decibels
- **Clarity Scoring**: Evaluates speech clarity based on transcription confidence
- **Pause Detection**: Tracks silence-to-speech ratio

### AI Interviewer Agent (`interviewer.py`)
- **Autonomous Interviewing**: Conducts mock interviews with natural conversation flow
- **Real-time Feedback**: Provides immediate feedback based on live metrics
- **Multiple Personas**: Three interviewer profiles with varying difficulty levels:
  - **Friendly Recruiter** (Easy): Patient, encouraging, provides hints
  - **Technical Lead** (Medium): Values clarity and precision
  - **Executive Director** (Hard): High standards, easily distracted, strict
- **Dynamic Interactions**: Reacts to posture and speech issues in real-time
- **Session Summaries**: Generates comprehensive feedback reports after interviews

### Frontend (`frontend/`)
- **Modern Web Interface**: Built with Next.js 16 and TypeScript
- **Real-time Data Display**: Shows live metrics and feedback
- **Video Stream Integration**: Displays live camera feed with zoom controls
- **Responsive Design**: Clean, modern UI with Tailwind CSS

## 🏗️ Architecture

The project follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐
│   Frontend      │  Next.js + TypeScript
│   (React)       │
└────────┬────────┘
         │
         │ HTTP/SSE
         │
┌────────▼─────────────────────────────────────┐
│         Backend Services                      │
├───────────────────────────────────────────────┤
│  camera.py  │  voice.py  │  interviewer.py   │
│  (CV)       │  (Audio)   │  (AI Agent)        │
└───────────────────────────────────────────────┘
         │
         │ JSON Data
         │
┌────────▼────────┐
│  MediaPipe      │  Computer Vision
│  Whisper        │  Speech Recognition
│  Gemini AI      │  LLM Agent
└─────────────────┘
```

## 📋 Prerequisites

- **Python 3.8+**
- **Node.js 18+** and npm
- **Webcam** for video analysis
- **Microphone** for audio analysis
- **Google Gemini API Key** (for AI interviewer)
- **OpenAI Whisper** (for speech transcription)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd video_meeting_analyzer
```

### 2. Backend Setup

Create two separate virtual environments for different dependencies:

#### Vision Environment (`.venv_vision`)
```bash
python -m venv .venv_vision
source .venv_vision/bin/activate  # On Windows: .venv_vision\Scripts\activate
pip install opencv-python numpy mediapipe pyaudio
```

**Note for macOS**: Install PortAudio first:
```bash
brew install portaudio
```

#### Agent Environment (`.venv_agent`)
```bash
python -m venv .venv_agent
source .venv_agent/bin/activate  # On Windows: .venv_agent\Scripts\activate
pip install google-generativeai python-dotenv
```

#### Install OpenAI Whisper
```bash
# In .venv_vision
pip install openai-whisper
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

## 📖 Usage

### Running the Backend Components

#### 1. Camera Analysis (Posture Detection)
```bash
# Activate vision environment
source .venv_vision/bin/activate  # On Windows: .venv_vision\Scripts\activate
python camera.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save current metrics

#### 2. Voice Analysis (Speech Detection)
```bash
# In vision environment (has Whisper)
python voice.py
```

**Features:**
- Analyzes speech in 5-second chunks
- Displays transcription, WPM, volume, and clarity
- Press `Ctrl+C` to stop

#### 3. AI Interviewer Agent
```bash
# Activate agent environment
source .venv_agent/bin/activate  # On Windows: .venv_agent\Scripts\activate
python interviewer.py
```

**Available Profiles:**
- `friendly_recruiter` - Easy difficulty
- `technical_lead` - Medium difficulty
- `tough_executive` - Hard difficulty

**Example:**
```python
from interviewer import AIInterviewerAgent

agent = AIInterviewerAgent(
    api_key="your_api_key",
    profile_key="friendly_recruiter"
)

# Start interview
opening = agent.start_interview()
print(opening)

# Process response with metrics
response = agent.process_response(
    candidate_response="Your answer here...",
    posture_summary={...},
    speech_summary={...}
)
```

### Running the Frontend

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📁 Project Structure

```
video_meeting_analyzer/
├── camera.py                 # Computer vision and posture analysis
├── voice.py                  # Speech analysis with Whisper
├── interviewer.py            # AI interviewer agent
├── agentTest.py              # Agent testing script
├── interview_agent/
│   ├── __init__.py
│   └── agent.py              # Google ADK agent implementation
├── frontend/
│   ├── app/
│   │   ├── page.tsx          # Landing page
│   │   ├── app/
│   │   │   └── page.tsx      # Main application page
│   │   ├── components/
│   │   │   └── Node.tsx      # Data display component
│   │   ├── layout.tsx        # Root layout
│   │   └── globals.css       # Global styles
│   ├── package.json
│   └── tsconfig.json
├── temp/
│   ├── static/              # Legacy static files
│   └── templates/           # Legacy templates
├── requirements.txt         # Python dependencies (reference)
├── tasks.txt                # Project tasks and notes
├── llmContext.txt           # Project context documentation
└── README.md                # This file
```

## 🔧 Technologies Used

### Backend
- **OpenCV**: Computer vision and video processing
- **MediaPipe**: Pose, hand, and face landmark detection
- **OpenAI Whisper**: Speech-to-text transcription
- **Google Gemini**: Large language model for AI interviewer
- **NumPy**: Numerical computations
- **PyAudio**: Audio input/output

### Frontend
- **Next.js 16**: React framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **React 19**: UI library

## 📊 Metrics Tracked

### Visual Metrics
- Shoulder angle (degrees)
- Head tilt (degrees)
- Forward lean (ratio)
- Head motion (pixels/frame)
- Hand motion (pixels/frame)
- Eye contact duration (seconds)
- Iris position (relative)

### Audio Metrics
- Words per minute (WPM)
- Volume (decibels)
- Clarity score (0-1)
- Pause ratio (0-1)
- Transcription text

## 🎯 Current Status (MVP)

- ✅ Computer vision analysis fully functional
- ✅ Speech analysis fully functional
- ✅ AI interviewer agent with multiple personas
- ✅ Real-time metric tracking
- ✅ Frontend interface
- 🔄 Integration between components (in progress)
- 🔄 Session tracking and scoring (planned)
- 🔄 User accounts (planned)

## 🚧 Future Enhancements

- **Additional Interviewer Personas**: More diverse interviewer styles
- **Session Tracking**: Save and review past interview sessions
- **User Accounts**: Personal progress tracking
- **Advanced Metrics**: 
  - Confidence estimation
  - Filler word detection
  - Sentiment analysis
- **Performance Scoring**: 0-100 scoring system
- **Customizable Thresholds**: Adjust sensitivity for different users
- **Export Reports**: Generate PDF reports of interview sessions


## 🙏 Acknowledgments

- **MediaPipe** for pose and face detection
- **OpenAI Whisper** for speech recognition
- **Google Gemini** for AI capabilities
- **Next.js** team for the excellent framework


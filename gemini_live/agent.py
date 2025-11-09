"""
agent.py - AI Interview Coach with Gemini Live Integration
Modified to support both batch analysis and real-time streaming
"""

from dotenv import load_dotenv
import os
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import json
from typing import Dict, Any, AsyncGenerator
import asyncio

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Check your .env file.")

os.environ["GOOGLE_API_KEY"] = api_key

# Quick connection test (optional)
client = genai.Client(api_key=api_key)
print("✅ Connected to Gemini")

# --- Constants ---
APP_NAME = "agent"
USER_ID = "test_user_interview"
SESSION_ID = "session_interview_001"
MODEL_NAME = "gemini-2.0-flash-exp"

# ============================================================================
# INPUT SCHEMA - Define what data the agent expects
# ============================================================================

class InterviewInput(BaseModel):
    """Input schema for interview agent"""
    message: str = Field(description="The message or metrics data to send to the interview coach")
    camera_data: Dict[str, Any] | None = Field(default=None, description="Optional camera metrics")
    voice_data: Dict[str, Any] | None = Field(default=None, description="Optional voice metrics")
    question_num: int | None = Field(default=None, description="Current question number")


# ============================================================================
# TOOL FUNCTIONS - These analyze metrics and return scores/feedback
# ============================================================================

def analyze_eye_contact(left_iris: float, right_iris: float, maintained: bool) -> dict:
    """Analyze eye contact quality based on iris position"""
    LEFT_EYE_MIN = 2.75
    LEFT_EYE_MAX = 2.85
    RIGHT_EYE_MIN = -1.95
    RIGHT_EYE_MAX = -1.875
    
    left_in_range = LEFT_EYE_MIN <= left_iris <= LEFT_EYE_MAX
    right_in_range = RIGHT_EYE_MIN <= right_iris <= RIGHT_EYE_MAX
    
    if not maintained:
        return {
            "score": 65,
            "feedback": "Try to maintain more consistent eye contact with the interviewer",
            "severity": "moderate"
        }
    elif left_in_range and right_in_range:
        return {
            "score": 95,
            "feedback": None,
            "severity": "none"
        }
    elif left_in_range or right_in_range:
        return {
            "score": 80,
            "feedback": "Your eye contact is pretty good, just keep it steady",
            "severity": "minor"
        }
    else:
        left_deviation = min(abs(left_iris - LEFT_EYE_MIN), abs(left_iris - LEFT_EYE_MAX))
        right_deviation = min(abs(right_iris - RIGHT_EYE_MIN), abs(right_iris - RIGHT_EYE_MAX))
        
        if left_deviation > 0.2 or right_deviation > 0.2:
            return {
                "score": 70,
                "feedback": "Try to focus more directly on the camera - your gaze is wandering a bit",
                "severity": "moderate"
            }
        else:
            return {
                "score": 85,
                "feedback": "Your eye contact is good, just try to keep it centered",
                "severity": "minor"
            }


def analyze_posture(shoulder_angle: float, head_tilt: float, forward_lean: float) -> dict:
    """Analyze overall posture quality"""
    SHOULDER_ANGLE_MIN = 165
    SHOULDER_ANGLE_MAX = 195
    HEAD_TILT_MIN = 165
    HEAD_TILT_MAX = 195
    FORWARD_LEAN_THRESHOLD = 0.15
    
    feedback_items = []
    scores = []
    
    if SHOULDER_ANGLE_MIN <= shoulder_angle <= SHOULDER_ANGLE_MAX:
        shoulder_deviation = abs(shoulder_angle - 180)
        scores.append(95 if shoulder_deviation <= 5 else 85)
    else:
        scores.append(70)
        feedback_items.append("keep your shoulders level and relaxed")
    
    if HEAD_TILT_MIN <= head_tilt <= HEAD_TILT_MAX:
        head_deviation = abs(head_tilt - 180)
        scores.append(95 if head_deviation <= 5 else 85)
    else:
        scores.append(70)
        feedback_items.append("keep your head straight")
    
    if forward_lean <= 0.10:
        scores.append(95)
    elif forward_lean <= FORWARD_LEAN_THRESHOLD:
        scores.append(85)
    else:
        scores.append(70)
        feedback_items.append("sit back a bit, you're leaning forward quite a lot")
    
    avg_score = sum(scores) / len(scores)
    
    if feedback_items:
        feedback = "For your posture, try to " + " and ".join(feedback_items)
        severity = "moderate" if avg_score < 80 else "minor"
    else:
        feedback = None
        severity = "none"
    
    return {
        "score": avg_score,
        "feedback": feedback,
        "severity": severity
    }


def analyze_movement(head_motion: float, hand_motion: float) -> dict:
    """Analyze body movement and fidgeting"""
    HEAD_MOTION_THRESHOLD = 15.0
    HAND_MOTION_THRESHOLD = 20.0
    
    feedback_items = []
    
    if head_motion < HEAD_MOTION_THRESHOLD * 0.5:
        head_score = 95
    elif head_motion < HEAD_MOTION_THRESHOLD:
        head_score = 90
    elif head_motion < HEAD_MOTION_THRESHOLD * 1.5:
        head_score = 75
        feedback_items.append("reduce head movement slightly")
    else:
        head_score = 65
        feedback_items.append("minimize head movement - try to keep your head steady")
    
    if hand_motion < HAND_MOTION_THRESHOLD * 0.5:
        hand_score = 95
    elif hand_motion < HAND_MOTION_THRESHOLD:
        hand_score = 90
    elif hand_motion < HAND_MOTION_THRESHOLD * 1.5:
        hand_score = 75
        feedback_items.append("try to reduce hand fidgeting")
    else:
        hand_score = 65
        feedback_items.append("your hand movements are quite distracting - keep them still or use purposeful gestures")
    
    avg_score = (head_score + hand_score) / 2
    
    if feedback_items:
        feedback = "Try to " + " and ".join(feedback_items)
        severity = "moderate" if avg_score < 75 else "minor"
    else:
        feedback = None
        severity = "none"
    
    return {
        "score": avg_score,
        "feedback": feedback,
        "severity": severity
    }


def analyze_speech_pace(words_per_minute: float) -> dict:
    """Analyze speaking pace"""
    MIN_WPM = 120
    MAX_WPM = 180
    IDEAL_WPM_MIN = 130
    IDEAL_WPM_MAX = 160
    
    if IDEAL_WPM_MIN <= words_per_minute <= IDEAL_WPM_MAX:
        return {"score": 95, "feedback": None, "severity": "none"}
    elif words_per_minute < MIN_WPM:
        if words_per_minute < MIN_WPM * 0.8:
            return {"score": 70, "feedback": "You can speak quite a bit faster - try to increase your pace", "severity": "moderate"}
        else:
            return {"score": 80, "feedback": "You can speak a bit faster - try to increase your pace slightly", "severity": "minor"}
    elif words_per_minute < IDEAL_WPM_MIN:
        return {"score": 85, "feedback": "Your pace is good, maybe pick up the speed just a little", "severity": "minor"}
    elif words_per_minute <= MAX_WPM:
        return {"score": 85, "feedback": "You're speaking a little fast - try to slow down and breathe", "severity": "minor"}
    else:
        return {"score": 70, "feedback": "You're speaking quite fast - take your time and pause between thoughts", "severity": "moderate"}


def analyze_volume(volume_db: float) -> dict:
    """Analyze speaking volume"""
    MIN_VOLUME_DB = -60
    
    if volume_db >= -50:
        return {"score": 95, "feedback": None, "severity": "none"}
    elif volume_db >= MIN_VOLUME_DB:
        return {"score": 85, "feedback": "Your volume is good, just try to project a little more", "severity": "minor"}
    elif volume_db >= -65:
        return {"score": 75, "feedback": "Try to speak up a bit - your volume is on the quiet side", "severity": "moderate"}
    else:
        return {"score": 65, "feedback": "Try to speak up - your volume is quite low", "severity": "moderate"}


def analyze_clarity(clarity_score: float) -> dict:
    """Analyze speech clarity"""
    MIN_CLARITY_SCORE = 0.7
    
    if clarity_score >= 0.9:
        return {"score": 95, "feedback": None, "severity": "none"}
    elif clarity_score >= 0.8:
        return {"score": 90, "feedback": None, "severity": "none"}
    elif clarity_score >= MIN_CLARITY_SCORE:
        return {"score": 80, "feedback": "Try to enunciate a bit more clearly", "severity": "minor"}
    else:
        return {"score": 70, "feedback": "Focus on speaking more clearly and enunciating your words", "severity": "moderate"}


def evaluate_overall_performance(question_num: int, metrics_summary: str) -> dict:
    """Generate overall performance evaluation for the interview"""
    if isinstance(metrics_summary, str):
        try:
            metrics = json.loads(metrics_summary)
        except:
            metrics = {"overall_score": 80}
    else:
        metrics = metrics_summary
    
    overall_score = metrics.get("overall_score", 80)
    
    if overall_score >= 90:
        performance_level = "Excellent"
        summary = "You demonstrated strong interview skills across all areas"
    elif overall_score >= 80:
        performance_level = "Good"
        summary = "You showed solid interview skills with room for minor improvements"
    elif overall_score >= 70:
        performance_level = "Fair"
        summary = "You have a good foundation but there are several areas to work on"
    else:
        performance_level = "Needs Improvement"
        summary = "Focus on the feedback provided to strengthen your interview presence"
    
    return {
        "performance_level": performance_level,
        "overall_score": overall_score,
        "summary": summary,
        "questions_completed": question_num
    }


# ============================================================================
# GEMINI LIVE INTEGRATION
# ============================================================================

class LiveInterviewCoach:
    """Real-time interview coach using Gemini Live API"""
    
    def __init__(self):
        self.client = genai.Client(api_key=api_key)
        self.config = types.LiveConnectConfig(
            response_modalities=["TEXT"],  # Text-only responses for now
        )
        
        self.system_instruction = """You are Sarah, a supportive real-time interview coach.

REAL-TIME MONITORING MODE:
You continuously receive metrics updates every few seconds during the interview response.
Your job is to monitor performance and provide gentle, immediate feedback when issues arise.

FEEDBACK RULES:
- Only give feedback when you see PERSISTENT problems (3+ consecutive updates)
- Keep feedback VERY brief and actionable (1 sentence max)
- Use gentle language: "Try to...", "Consider...", "Remember to..."
- Don't interrupt for minor issues
- Focus on the most critical issue at a time

METRICS YOU MONITOR:
- Eye contact (iris position and gaze maintenance)
- Posture (shoulders, head tilt, forward lean)
- Movement (head motion, hand fidgeting)
- Speech (pace, volume, clarity) - when available

EXAMPLE INTERVENTIONS:
- "Try to look directly at the camera"
- "Sit up a bit straighter"
- "Try to keep your hands still"
- "Slow down your pace a little"

Remember: Your goal is to help without being distracting. Only speak up when really needed."""
    
    async def stream_analysis(
        self, 
        camera_stream: AsyncGenerator[Dict, None],
        voice_stream: AsyncGenerator[Dict, None] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream real-time analysis as metrics come in
        
        Args:
            camera_stream: Async generator yielding camera metrics
            voice_stream: Async generator yielding voice metrics (optional)
            
        Yields:
            Real-time feedback messages from the coach
        """
        session = self.client.aio.live.connect(
            model=MODEL_NAME,
            config=self.config
        )
        
        async with session:
            # Send system instruction
            await session.send(self.system_instruction, end_of_turn=True)
            
            # Process camera metrics as they arrive
            async for camera_data in camera_stream:
                # Format metrics for the model
                metrics_text = self._format_metrics_for_live(camera_data)
                
                # Send to model
                await session.send(metrics_text, end_of_turn=True)
                
                # Get response
                async for response in session.receive():
                    if response.text:
                        yield response.text
    
    def _format_metrics_for_live(self, metrics: Dict) -> str:
        """Format metrics in a concise way for real-time streaming"""
        return f"""Current metrics:
- Eye contact: {metrics.get('eye_contact_maintained', True)}
- Iris: L={metrics.get('left_iris_relative', 0):.2f}, R={metrics.get('right_iris_relative', 0):.2f}
- Posture: shoulder={metrics.get('shoulder_angle', 180):.1f}°, tilt={metrics.get('head_tilt', 180):.1f}°, lean={metrics.get('forward_lean', 0):.2f}
- Motion: head={metrics.get('head_motion_score', 0):.1f}, hands={metrics.get('hand_motion_score', 0):.1f}"""


# ============================================================================
# AGENT DEFINITION (Original batch processing)
# ============================================================================

interview_agent = LlmAgent(
    model=MODEL_NAME,
    name="Sarah",
    description="A friendly, supportive interview coach who helps candidates practice and improve their interview skills",
    
    instruction="""You are Sarah, a warm and encouraging interview coach conducting a mock interview practice session.

YOUR ROLE:
- Conduct a 2-question mock interview
- Analyze the candidate's posture, eye contact, speech pace, and volume
- Provide constructive feedback after each answer
- Maintain a friendly, casual tone throughout
- End with an overall performance summary

INTERVIEW FLOW:
1. When you receive "Hello! I'm ready to start my practice interview.", greet the candidate warmly and ask your first question. Keep it conversational, like: "Tell me a bit about yourself and what brings you here today."

2. When you receive metrics analysis (Camera Metrics and Voice Metrics), this means the candidate has finished answering. You should:
   a) Use the analysis tools to evaluate their performance
   b) Give specific, actionable feedback in a friendly way
   c) If this is Question 1, ask your second question: "What do you think is your greatest strength and how has it helped you in the past?"
   d) If this is Question 2, provide a final summary with overall scores

FEEDBACK STYLE:
- Start with positive reinforcement (e.g., "Great job!" or "Nice work!")
- Be specific: mention exact behaviors
- Keep feedback conversational and supportive
- Limit feedback to 2-3 key points per response
- Use phrases like "try to", "consider", "it might help to" rather than commands

FINAL SUMMARY (after Question 2):
- Congratulate them on completing the practice session
- Provide overall scores for: Engagement, Clarity, and Posture
- Give 2-3 concrete next steps for improvement
- End with encouragement

IMPORTANT:
- Always call the analysis tools when you receive metrics
- Base your feedback on the tool results
- Keep your tone warm, never harsh or discouraging
- Remember this is practice - the goal is to help them improve""",
    
    tools=[
        analyze_eye_contact,
        analyze_posture,
        analyze_movement,
        analyze_speech_pace,
        analyze_volume,
        analyze_clarity,
        evaluate_overall_performance
    ],
    
    input_schema=InterviewInput,
    output_key="interview_response"
)

# ============================================================================
# SESSION AND RUNNER SETUP
# ============================================================================

session_service = InMemorySessionService()
interview_runner = Runner(
    agent=interview_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# Create live coach instance
live_coach = LiveInterviewCoach()

# ============================================================================
# EXPORT FOR APP.PY
# ============================================================================

__all__ = [
    "APP_NAME",
    "USER_ID",
    "SESSION_ID",
    "interview_runner",
    "interview_agent",
    "session_service",
    "types",
    "InterviewInput",
    "live_coach"  # NEW: Export live coach
]
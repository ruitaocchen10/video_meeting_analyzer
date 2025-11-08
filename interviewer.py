import google.generativeai as genai
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import threading
import queue
from datetime import datetime
import random

@dataclass
class InterviewerProfile:
    """Different interviewer personalities with varying tolerances"""
    name: str
    difficulty: str  # "Easy", "Medium", "Hard"
    description: str
    
    # Tolerance thresholds
    max_wpm: int
    min_wpm: int
    min_volume_db: float
    min_clarity: float
    max_head_motion: float
    max_hand_motion: float
    eye_contact_threshold: float
    max_forward_lean: float
    
    # Behavioral traits
    interrupts_frequently: bool
    asks_followups: bool
    provides_hints: bool
    patience_level: str  # "Low", "Medium", "High"
    
    # Question style
    question_types: List[str]  # ["behavioral", "technical", "situational", "curveball"]

# Predefined interviewer profiles
INTERVIEWER_PROFILES = {
    "friendly_recruiter": InterviewerProfile(
        name="Sarah - Friendly Recruiter",
        difficulty="Easy",
        description="A warm, supportive recruiter focused on getting to know you. Very patient and encouraging.",
        max_wpm=200,
        min_wpm=100,
        min_volume_db=-70,
        min_clarity=0.6,
        max_head_motion=25.0,
        max_hand_motion=30.0,
        eye_contact_threshold=3.0,
        max_forward_lean=0.2,
        interrupts_frequently=False,
        asks_followups=True,
        provides_hints=True,
        patience_level="High",
        question_types=["behavioral", "situational"]
    ),
    "technical_lead": InterviewerProfile(
        name="Alex - Technical Lead",
        difficulty="Medium",
        description="An experienced engineer who values clarity and precision. Expects confident, well-structured answers.",
        max_wpm=180,
        min_wpm=120,
        min_volume_db=-50,
        min_clarity=0.75,
        max_head_motion=15.0,
        max_hand_motion=20.0,
        eye_contact_threshold=2.0,
        max_forward_lean=0.15,
        interrupts_frequently=False,
        asks_followups=True,
        provides_hints=False,
        patience_level="Medium",
        question_types=["technical", "behavioral", "situational"]
    ),
    "tough_executive": InterviewerProfile(
        name="Morgan - Executive Director",
        difficulty="Hard",
        description="A no-nonsense executive with high standards. Easily distracted, values confidence and professionalism.",
        max_wpm=170,
        min_wpm=130,
        min_volume_db=-40,
        min_clarity=0.85,
        max_head_motion=10.0,
        max_hand_motion=15.0,
        eye_contact_threshold=1.5,
        max_forward_lean=0.1,
        interrupts_frequently=True,
        asks_followups=True,
        provides_hints=False,
        patience_level="Low",
        question_types=["behavioral", "situational", "curveball"]
    )
}

@dataclass
class PostureSnapshot:
    """Snapshot of posture metrics"""
    shoulder_angle: float
    head_tilt: float
    forward_lean: float
    head_motion: float
    hand_motion: float
    eye_contact_maintained: bool
    timestamp: float

@dataclass
class SpeechSnapshot:
    """Snapshot of speech metrics"""
    text: str
    words_per_minute: float
    volume_db: float
    clarity_score: float
    timestamp: float

@dataclass
class InteractionEvent:
    """Events that the agent can trigger"""
    event_type: str  # "interrupt", "visual_cue", "encouragement", "correction", "distraction"
    message: str
    timestamp: float
    severity: str  # "low", "medium", "high"

class AIInterviewerAgent:
    """Advanced AI interviewer with real-time monitoring and interaction"""
    
    def __init__(self, api_key: str, profile_key: str = "friendly_recruiter"):
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.chat = None
        
        # Load interviewer profile
        self.profile = INTERVIEWER_PROFILES[profile_key]
        
        # Monitoring queues
        self.posture_queue = queue.Queue()
        self.speech_queue = queue.Queue()
        self.event_queue = queue.Queue()
        
        # Session tracking
        self.current_question = None
        self.question_count = 0
        self.session_start_time = None
        self.conversation_history = []
        self.performance_log = []
        
        # Real-time monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Behavioral state
        self.distraction_level = 0  # 0-100, increases with poor posture
        self.engagement_level = 100  # 0-100, decreases with issues
        self.last_interruption_time = 0
        
        # Initialize chat with system prompt
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the AI agent with detailed system prompt"""
        system_prompt = f"""You are {self.profile.name}, {self.profile.description}

Your role is to conduct a professional interview while monitoring the candidate's communication skills in real-time.

INTERVIEWER PERSONALITY:
- Difficulty: {self.profile.difficulty}
- Patience Level: {self.profile.patience_level}
- You {"DO" if self.profile.interrupts_frequently else "DON'T"} interrupt when you notice issues
- You {"DO" if self.profile.asks_followups else "DON'T"} ask follow-up questions
- You {"DO" if self.profile.provides_hints else "DON'T"} provide hints when candidates struggle

YOUR BEHAVIORAL STANDARDS:
- Speaking rate: {self.profile.min_wpm}-{self.profile.max_wpm} WPM (you notice when too fast/slow)
- Volume: Above {self.profile.min_volume_db} dB (you notice when too quiet)
- Clarity: Above {self.profile.min_clarity} (you notice unclear speech)
- Eye contact: You get distracted if they look away for >{self.profile.eye_contact_threshold}s
- Posture: You notice slouching, leaning, excessive movement

INTERACTION MODES:
1. QUESTIONING MODE: Ask interview questions appropriate to the role
2. MONITORING MODE: React to real-time metrics (posture, speech, eye contact)
3. FEEDBACK MODE: Provide constructive feedback after responses
4. INTERRUPTION MODE: Interrupt if issues are severe (based on your personality)
5. ENCOURAGEMENT MODE: Provide positive reinforcement for good performance

QUESTION TYPES YOU ASK: {', '.join(self.profile.question_types)}

IMPORTANT BEHAVIORAL RULES:
- React authentically to what you observe (getting distracted, confused, impressed)
- If eye contact is lost repeatedly, mention "I notice you're looking away..."
- If posture is poor, subtly comment "Are you comfortable? You seem to be leaning..."
- If speaking too fast, interrupt with "Hold on, could you slow down a bit?"
- If volume is low, say "Sorry, I'm having trouble hearing you..."
- Give specific, actionable feedback
- Stay in character throughout the interview
- Keep responses concise (2-4 sentences typically)
- Balance realism with helpfulness

You are conducting a LIVE interview. Respond naturally and dynamically based on the candidate's real-time performance."""

        self.conversation_history.append({
            "role": "system",
            "content": system_prompt
        })
    
    def start_interview(self):
        """Begin the interview session"""
        self.session_start_time = time.time()
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker)
        self.monitoring_thread.start()
        
        # Generate opening
        opening_prompt = f"""Start the interview with a warm greeting and your first question. 
        Remember you are {self.profile.name}. Make it natural and conversational."""
        
        response = self.model.generate_content(
            self._build_prompt_with_history(opening_prompt)
        )
        
        question = response.text
        self.current_question = question
        self.question_count += 1
        
        self._log_interaction("agent", question)
        
        return question
    
    def _build_prompt_with_history(self, new_prompt: str) -> str:
        """Build prompt with conversation history"""
        history_text = "\n\n".join([
            f"{'SYSTEM' if msg['role'] == 'system' else msg['role'].upper()}: {msg['content']}"
            for msg in self.conversation_history[-10:]  # Last 10 messages
        ])
        
        return f"{history_text}\n\nUSER: {new_prompt}\n\nAGENT:"
    
    def update_posture(self, posture_data: Dict):
        """Receive posture update from camera.py"""
        snapshot = PostureSnapshot(
            shoulder_angle=posture_data.get('shoulder_angle', 180),
            head_tilt=posture_data.get('head_tilt', 180),
            forward_lean=posture_data.get('forward_lean', 0),
            head_motion=posture_data.get('head_motion', 0),
            hand_motion=posture_data.get('hand_motion', 0),
            eye_contact_maintained=posture_data.get('eye_contact_maintained', True),
            timestamp=time.time()
        )
        self.posture_queue.put(snapshot)
    
    def update_speech(self, speech_data: Dict):
        """Receive speech update from voice.py"""
        snapshot = SpeechSnapshot(
            text=speech_data.get('text', ''),
            words_per_minute=speech_data.get('words_per_minute', 0),
            volume_db=speech_data.get('volume_db', -50),
            clarity_score=speech_data.get('clarity_score', 0.8),
            timestamp=time.time()
        )
        self.speech_queue.put(snapshot)
    
    def _monitoring_worker(self):
        """Background thread that monitors metrics and triggers events"""
        while self.is_monitoring:
            try:
                # Check for posture issues
                if not self.posture_queue.empty():
                    posture = self.posture_queue.get_nowait()
                    self._analyze_posture(posture)
                
                # Check for speech issues
                if not self.speech_queue.empty():
                    speech = self.speech_queue.get_nowait()
                    self._analyze_speech(speech)
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"Monitoring error: {e}")
    
    def _analyze_posture(self, posture: PostureSnapshot):
        """Analyze posture and trigger events if needed"""
        issues = []
        
        # Eye contact check
        if not posture.eye_contact_maintained:
            self.distraction_level += 5
            self.engagement_level -= 3
            if self.distraction_level > 30:
                issues.append(("eye_contact", "high"))
        else:
            self.distraction_level = max(0, self.distraction_level - 1)
        
        # Forward lean
        if posture.forward_lean > self.profile.max_forward_lean:
            issues.append(("posture", "medium"))
        
        # Excessive movement
        if posture.head_motion > self.profile.max_head_motion:
            self.distraction_level += 3
            issues.append(("head_motion", "medium"))
        
        if posture.hand_motion > self.profile.max_hand_motion:
            issues.append(("hand_fidgeting", "low"))
        
        # Trigger interruption if needed
        if issues and self._should_interrupt():
            self._trigger_interruption(issues)
    
    def _analyze_speech(self, speech: SpeechSnapshot):
        """Analyze speech and trigger events if needed"""
        issues = []
        
        # Speaking rate
        if speech.words_per_minute > self.profile.max_wpm:
            issues.append(("too_fast", "medium"))
        elif speech.words_per_minute < self.profile.min_wpm and speech.words_per_minute > 0:
            issues.append(("too_slow", "low"))
        
        # Volume
        if speech.volume_db < self.profile.min_volume_db:
            issues.append(("too_quiet", "high"))
        
        # Clarity
        if speech.clarity_score < self.profile.min_clarity:
            issues.append(("unclear", "medium"))
        
        if issues and self._should_interrupt():
            self._trigger_interruption(issues)
    
    def _should_interrupt(self) -> bool:
        """Decide whether to interrupt based on personality and timing"""
        if not self.profile.interrupts_frequently:
            return False
        
        # Don't interrupt too frequently
        if time.time() - self.last_interruption_time < 15:
            return False
        
        # Probability based on distraction and patience
        if self.profile.patience_level == "Low":
            return random.random() < 0.3
        elif self.profile.patience_level == "Medium":
            return random.random() < 0.15
        else:
            return random.random() < 0.05
    
    def _trigger_interruption(self, issues: List[tuple]):
        """Trigger an interruption event"""
        self.last_interruption_time = time.time()
        
        issue_type, severity = issues[0]
        
        interruptions = {
            "eye_contact": "Sorry, I notice you keep looking away. Everything okay?",
            "posture": "Are you comfortable? You seem to be leaning quite a bit.",
            "head_motion": "I notice you're moving around a lot. Are you okay?",
            "too_fast": "Hold on - you're speaking quite fast. Could you slow down a bit?",
            "too_quiet": "Sorry, I'm having trouble hearing you. Could you speak up?",
            "unclear": "I'm having trouble understanding. Could you repeat that more clearly?"
        }
        
        message = interruptions.get(issue_type, "Let me stop you there for a moment.")
        
        event = InteractionEvent(
            event_type="interrupt",
            message=message,
            timestamp=time.time(),
            severity=severity
        )
        
        self.event_queue.put(event)
        print(f"\n🔴 INTERRUPTION: {message}\n")
    
    def process_response(self, candidate_response: str, posture_summary: Dict, speech_summary: Dict) -> str:
        """Process candidate's response and generate next question or feedback"""
        
        # Build comprehensive context
        context = f"""
CANDIDATE RESPONSE: "{candidate_response}"

REAL-TIME METRICS DURING RESPONSE:
Posture:
- Shoulder angle: {posture_summary.get('shoulder_angle', 'N/A')}°
- Head tilt: {posture_summary.get('head_tilt', 'N/A')}°
- Forward lean: {posture_summary.get('forward_lean', 'N/A')}
- Head motion: {posture_summary.get('head_motion', 'N/A')} px/frame
- Hand motion: {posture_summary.get('hand_motion', 'N/A')} px/frame
- Eye contact: {'Maintained' if posture_summary.get('eye_contact_maintained', True) else 'Lost frequently'}

Speech:
- Speaking rate: {speech_summary.get('words_per_minute', 'N/A')} WPM (ideal: {self.profile.min_wpm}-{self.profile.max_wpm})
- Volume: {speech_summary.get('volume_db', 'N/A')} dB (minimum: {self.profile.min_volume_db})
- Clarity: {speech_summary.get('clarity_score', 'N/A')} (minimum: {self.profile.min_clarity})

YOUR CURRENT STATE:
- Distraction level: {self.distraction_level}/100
- Engagement level: {self.engagement_level}/100

Based on this, provide:
1. Your natural reaction to their response (impressed, confused, distracted, etc.)
2. Brief, specific feedback on communication issues you noticed
3. Either a follow-up question OR move to the next interview question

Stay in character as {self.profile.name}. Be authentic and react to what you observed."""

        # Log performance
        self.performance_log.append({
            'question': self.current_question,
            'response': candidate_response,
            'posture': posture_summary,
            'speech': speech_summary,
            'distraction_level': self.distraction_level,
            'engagement_level': self.engagement_level,
            'timestamp': time.time()
        })
        
        # Generate response
        response = self.model.generate_content(
            self._build_prompt_with_history(context)
        )
        
        agent_response = response.text
        self._log_interaction("candidate", candidate_response)
        self._log_interaction("agent", agent_response)
        
        return agent_response
    
    def _log_interaction(self, speaker: str, message: str):
        """Log conversation interaction"""
        self.conversation_history.append({
            "role": speaker,
            "content": message,
            "timestamp": time.time()
        })
    
    def get_realtime_event(self) -> Optional[InteractionEvent]:
        """Check if there's a real-time event to display"""
        try:
            return self.event_queue.get_nowait()
        except queue.Empty:
            return None
    
    def end_interview(self) -> Dict:
        """End the interview and generate comprehensive feedback"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        # Generate final evaluation
        evaluation_prompt = f"""
The interview is complete. Based on the entire session, provide a comprehensive evaluation:

SESSION SUMMARY:
- Questions asked: {self.question_count}
- Duration: {(time.time() - self.session_start_time) / 60:.1f} minutes
- Final distraction level: {self.distraction_level}/100
- Final engagement level: {self.engagement_level}/100

PERFORMANCE LOG:
{json.dumps(self.performance_log, indent=2)}

Provide:
1. Overall performance score (0-100)
2. Strengths (3-5 specific points)
3. Areas for improvement (3-5 specific points)
4. Specific recommendations for next practice session
5. One piece of encouraging advice

Format as JSON with keys: score, strengths, improvements, recommendations, encouragement
"""
        
        response = self.model.generate_content(evaluation_prompt)
        
        try:
            # Try to parse as JSON
            evaluation = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
        except:
            # Fallback to text
            evaluation = {
                "score": self.engagement_level,
                "raw_feedback": response.text
            }
        
        return {
            "evaluation": evaluation,
            "session_data": {
                "questions": self.question_count,
                "duration": time.time() - self.session_start_time,
                "conversation": self.conversation_history,
                "performance": self.performance_log
            }
        }
    
    def save_session(self, filename: str = None):
        """Save session data to file"""
        if filename is None:
            filename = f"interview_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            "interviewer": self.profile.name,
            "difficulty": self.profile.difficulty,
            "timestamp": datetime.now().isoformat(),
            "conversation": self.conversation_history,
            "performance": self.performance_log,
            "final_metrics": {
                "distraction_level": self.distraction_level,
                "engagement_level": self.engagement_level
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Session saved to {filename}")
        return filename


# Example integration function
def run_interview_session(gemini_api_key: str, profile: str = "friendly_recruiter"):
    """
    Example function showing how to integrate with camera.py and voice.py
    
    This would be called from a main coordinator script that:
    1. Runs camera.py in one thread
    2. Runs voice.py in another thread
    3. Feeds data to this agent
    4. Displays agent responses
    """
    
    # Initialize agent
    agent = AIInterviewerAgent(gemini_api_key, profile)
    
    # Start interview
    opening = agent.start_interview()
    print(f"\n{'='*60}")
    print(f"INTERVIEWER: {opening}")
    print(f"{'='*60}\n")
    
    # Example of processing a response
    # In real implementation, this would come from voice.py transcription
    example_response = "I have 5 years of experience in software development..."
    
    # Example metrics that would come from camera.py and voice.py
    posture_metrics = {
        'shoulder_angle': 178.0,
        'head_tilt': 182.0,
        'forward_lean': 0.08,
        'head_motion': 8.5,
        'hand_motion': 12.3,
        'eye_contact_maintained': True
    }
    
    speech_metrics = {
        'text': example_response,
        'words_per_minute': 145,
        'volume_db': -45,
        'clarity_score': 0.82
    }
    
    # Update agent with real-time data
    agent.update_posture(posture_metrics)
    agent.update_speech(speech_metrics)
    
    # Process response
    next_question = agent.process_response(example_response, posture_metrics, speech_metrics)
    print(f"\nINTERVIEWER: {next_question}\n")
    
    # Check for real-time events
    event = agent.get_realtime_event()
    if event:
        print(f"⚠️  REAL-TIME FEEDBACK: {event.message}")
    
    return agent


if __name__ == "__main__":
    # Demo with mock API key
    print("AI Interviewer Agent")
    print("="*60)
    print("\nAvailable interviewer profiles:")
    for key, profile in INTERVIEWER_PROFILES.items():
        print(f"\n{key}:")
        print(f"  {profile.name}")
        print(f"  Difficulty: {profile.difficulty}")
        print(f"  {profile.description}")
    
    print("\n" + "="*60)
    print("\nTo use this agent, you need to:")
    print("1. Get a Google Gemini API key from https://makersuite.google.com/app/apikey")
    print("2. Run: agent = AIInterviewerAgent(api_key='YOUR_KEY', profile_key='friendly_recruiter')")
    print("3. Integrate with camera.py and voice.py to feed real-time data")
    print("4. The agent will monitor, react, and provide dynamic feedback")
    print("\n" + "="*60)
"""
app.py - Integrated Camera + Voice data collection with agent feedback
Camera runs in MAIN thread (OpenCV requirement), Voice runs in background thread
"""

import asyncio
import json
import threading
import queue
import time
from camera import stream_camera_metrics
from voice import stream_voice_with_text_vad
from interview_agent.agent import (
    APP_NAME,
    USER_ID,
    SESSION_ID,
    interview_runner,
    interview_agent,
    session_service,
    types
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MINIMUM_RECORDING_TIME = 10.0  # Minimum seconds before allowing collection to stop


def aggregate_camera_metrics(all_metrics):
    """Calculate average camera metrics from collected data"""
    if not all_metrics:
        return {}
    
    total_frames = len(all_metrics)
    
    avg_metrics = {
        "shoulder_angle": sum(d.get('shoulder_angle', 180) for d in all_metrics) / total_frames,
        "head_tilt": sum(d.get('head_tilt', 180) for d in all_metrics) / total_frames,
        "forward_lean": sum(d.get('forward_lean', 0) for d in all_metrics) / total_frames,
        "head_motion": sum(d.get('head_motion_score', 0) for d in all_metrics) / total_frames,
        "hand_motion": sum(d.get('hand_motion_score', 0) for d in all_metrics) / total_frames,
    }
    
    # Calculate eye contact percentage
    eye_contact_frames = sum(1 for d in all_metrics if d.get('eye_contact_maintained', True))
    avg_metrics['eye_contact_percentage'] = (eye_contact_frames / total_frames) * 100
    
    # Get most recent iris positions
    last_frame = all_metrics[-1]
    avg_metrics['left_iris_relative'] = last_frame.get('left_iris_relative')
    avg_metrics['right_iris_relative'] = last_frame.get('right_iris_relative')
    avg_metrics['eye_contact_maintained'] = last_frame.get('eye_contact_maintained', True)
    
    # Count issues
    all_issues = []
    for d in all_metrics:
        all_issues.extend(d.get('issues', []))
    
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    avg_metrics['issue_summary'] = issue_counts
    avg_metrics['total_frames'] = total_frames
    
    return avg_metrics


def format_camera_metrics(camera_data):
    """Format camera metrics into readable string for agent"""
    return f"""Camera Metrics (Averaged over {camera_data.get('total_frames', 0)} frames):
- Eye Contact: {camera_data.get('eye_contact_percentage', 0):.1f}% maintained
- Left Iris Position: {camera_data.get('left_iris_relative', 0):.3f} (valid: 2.75-2.95)
- Right Iris Position: {camera_data.get('right_iris_relative', 0):.3f} (valid: -1.95 to -1.775)
- Shoulder Angle: {camera_data.get('shoulder_angle', 180):.1f}° (acceptable: 165-195°)
- Head Tilt: {camera_data.get('head_tilt', 180):.1f}° (acceptable: 165-195°)
- Forward Lean: {camera_data.get('forward_lean', 0.0):.3f} (threshold: 0.15)
- Head Motion: {camera_data.get('head_motion', 0):.1f} px/frame (threshold: 15.0)
- Hand Motion: {camera_data.get('hand_motion', 0):.1f} px/frame (threshold: 20.0)
- Issues Detected: {camera_data.get('issue_summary', {})}"""


def format_speech_metrics(speech_data):
    """Format speech metrics into readable string for agent"""
    if not speech_data:
        return "Speech Metrics: No speech detected"
    
    return f"""Speech Metrics:
- Transcript: "{speech_data.get('text', 'N/A')}"
- Duration: {speech_data.get('speech_duration', 0):.1f}s
- Words Per Minute: {speech_data.get('words_per_minute', 0):.0f}
- Volume: {speech_data.get('volume_db', 0):.1f} dB
- Clarity Score: {speech_data.get('clarity_score', 0):.2f}"""


def voice_collector(voice_queue, min_recording_time):
    """Thread function to collect voice metrics - runs in BACKGROUND"""
    speech_metrics = None
    start_time = time.time()
    
    try:
        print(f"🎤 Voice collector started (minimum {min_recording_time}s)...")
        for data in stream_voice_with_text_vad(no_speech_duration=2.0, transcribe_interval=0.5):
            
            if data["type"] == "speech_started":
                print("\n🎤 Speech detected! Recording...")
            
            elif data["type"] == "status":
                current_text = data.get("text", "")
                display_text = current_text if len(current_text) <= 50 else current_text[:47] + "..."
                elapsed = time.time() - start_time
                print(f"\r🔴 [{elapsed:.1f}s] Speaking: {display_text:<50}", end="", flush=True)
            
            elif data["type"] == "speech_complete":
                elapsed = time.time() - start_time
                
                # Check if minimum time has elapsed
                if elapsed >= min_recording_time:
                    speech_metrics = data["metrics"]
                    print(f"\n✅ Speech complete! (Duration: {elapsed:.1f}s)")
                    break
                else:
                    remaining = min_recording_time - elapsed
                    print(f"\n⏳ Minimum time not reached. Continuing for {remaining:.1f}s more...")
                    # Continue collecting - don't break yet
    
    except Exception as e:
        print(f"\nVoice collector error: {e}")
    finally:
        voice_queue.put(speech_metrics)
        print("🎤 Voice collector stopped")


async def send_to_agent(camera_metrics, speech_metrics):
    """Send combined metrics to agent and get feedback"""
    # Format message for agent
    message = f"""Interview Response Analysis:

{format_camera_metrics(camera_metrics)}

{format_speech_metrics(speech_metrics)}

Please analyze these metrics and provide comprehensive feedback on:
1. Posture and body language
2. Eye contact
3. Speech delivery (pace, clarity, volume)
4. Overall presentation

Then, ask the next interview question."""
    
    print(f"\n{'='*70}")
    print(">>> Sending to Agent...")
    print(f"{'='*70}")
    
    # Create user content
    user_content = types.Content(
        role='user',
        parts=[types.Part(text=message)]
    )
    
    # Get agent response
    final_response = "No response received."
    async for event in interview_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
    
    print(f"\n<<< Agent Feedback:")
    print(final_response)
    print(f"{'='*70}\n")


async def collect_and_analyze():
    """Collect camera and voice data - camera in MAIN thread, voice in BACKGROUND"""
    
    # Create queue for voice data
    voice_queue = queue.Queue()
    
    # Start voice collector in background thread with minimum recording time
    voice_thread = threading.Thread(
        target=voice_collector,
        args=(voice_queue, MINIMUM_RECORDING_TIME),
        daemon=True
    )
    voice_thread.start()
    
    # Collect camera data in MAIN thread (EXACT same as original tempApp.py)
    all_camera_metrics = []
    
    print("\n" + "="*70)
    print("📊 DATA COLLECTION")
    print("="*70)
    print(f"Minimum recording time: {MINIMUM_RECORDING_TIME}s")
    print("Starting camera metrics collection...")
    print("Voice collector running in background...")
    print("Press 'q' in the video window to stop manually.")
    print("="*70 + "\n")
    
    try:
        # Stream metrics from camera.py - RUNS IN MAIN THREAD
        for metrics in stream_camera_metrics(show_video=True):
            all_camera_metrics.append(metrics)
            
            # Check if voice thread is still alive
            if not voice_thread.is_alive():
                # Voice collector finished - stop camera too
                print("\n🎤 Voice collection complete - stopping camera...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    
    # Wait for voice thread to complete (if not already done)
    voice_thread.join(timeout=1.0)
    
    # Get voice data from queue
    speech_metrics = None
    try:
        speech_metrics = voice_queue.get_nowait()
    except queue.Empty:
        print("⚠️  No speech data collected")
    
    # Data collection finished - process and send to agent
    print("\n" + "="*70)
    print(f"✅ Collection complete! Total frames: {len(all_camera_metrics)}")
    print(f"   Speech: {'Detected' if speech_metrics else 'Not detected'}")
    print("="*70)
    
    if all_camera_metrics:
        # Aggregate camera metrics
        camera_metrics = aggregate_camera_metrics(all_camera_metrics)
        
        print("\n" + "="*70)
        print("📊 AGGREGATED METRICS")
        print("="*70)
        print(format_camera_metrics(camera_metrics))
        print()
        print(format_speech_metrics(speech_metrics))
        print("="*70)
        
        # Send to agent
        await send_to_agent(camera_metrics, speech_metrics)
    else:
        print("\n⚠️  No camera data collected!")


async def main():
    """Main function - runs interview loop with integrated data collection"""
    
    # Setup agent session
    print("\n" + "="*70)
    print("🎯 INTEGRATED INTERVIEW PRACTICE")
    print("   Camera (Main Thread) + Voice (Background Thread)")
    print(f"   Minimum Recording Time: {MINIMUM_RECORDING_TIME}s")
    print("="*70)
    
    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        print(f"✅ Session created: {SESSION_ID}\n")
    except Exception as e:
        print(f"Session note: {e}\n")
    
    # Start interview
    print("🚀 Starting interview...")
    user_content = types.Content(
        role='user',
        parts=[types.Part(text="Hello! I'm ready to start my practice interview.")]
    )
    async for event in interview_runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(f"\nAgent: {event.content.parts[0].text}\n")
    
    # Interview loop
    while True:
        print("\n" + "="*70)
        ready = input("Ready to answer? (y/n or 'quit' to exit): ").strip().lower()
        
        if ready == 'quit':
            print("\n👋 Ending interview session...")
            break
        
        if ready != 'y':
            continue
        
        # Collect and analyze response
        await collect_and_analyze()
        
        # Ask if user wants to continue
        print("\n" + "="*70)
        continue_interview = input("Continue with next question? (y/n): ").strip().lower()
        
        if continue_interview != 'y':
            print("\n👋 Ending interview session...")
            break
    
    print("\n" + "="*70)
    print("✅ INTERVIEW COMPLETE")
    print("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
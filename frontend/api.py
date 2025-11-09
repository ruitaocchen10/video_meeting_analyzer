"""
FastAPI + Socket.IO backend for real-time video/audio processing
Maintains persistent connection with frontend for bidirectional communication
"""

import asyncio
import base64
import cv2
import numpy as np
import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import sys
import os

# Add parent directory to path to import camera and voice modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera import PostureDetector

# Initialize Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',  # Allow all origins (restrict in production)
    logger=False,  # Disable verbose Socket.IO logging
    engineio_logger=False  # Disable verbose Engine.IO logging
)

# Initialize FastAPI
app = FastAPI(title="Video Meeting Analyzer API")

# Add CORS middleware for HTTP endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wrap FastAPI app with Socket.IO
socket_app = socketio.ASGIApp(
    sio,
    app,
    socketio_path='/socket.io'
)

# Global state for camera and processing
posture_detector = None
camera_active = False
video_capture = None
current_frame = None
current_metrics = None


# ============================================================================
# SOCKET.IO EVENT HANDLERS
# ============================================================================

@sio.event
async def connect(sid, environ):
    """Called when a client connects"""
    print(f"Client connected: {sid}")
    await sio.emit('connection_status', {'status': 'connected', 'message': 'Connected to backend'}, to=sid)


@sio.event
async def disconnect(sid):
    """Called when a client disconnects"""
    print(f"Client disconnected: {sid}")
    global camera_active
    camera_active = False


@sio.event
async def start_camera(sid, data):
    """Start camera processing and streaming"""
    global camera_active, posture_detector, video_capture

    print(f"[DEBUG] Client {sid} requested to start camera")

    if camera_active:
        print(f"[DEBUG] Camera already active, rejecting request")
        await sio.emit('error', {'message': 'Camera already active'}, to=sid)
        return

    try:
        # Initialize posture detector
        print(f"[DEBUG] Initializing posture detector...")
        if posture_detector is None:
            posture_detector = PostureDetector()
        print(f"[DEBUG] Posture detector ready")

        # Initialize camera
        print(f"[DEBUG] Opening camera device 0...")
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            raise Exception("Failed to open camera device 0")

        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Test read to ensure camera works
        success, test_frame = video_capture.read()
        if not success:
            raise Exception("Camera opened but failed to read frame")

        print(f"[DEBUG] Camera opened successfully, resolution: {test_frame.shape}")

        camera_active = True
        print(f"[DEBUG] Camera active flag set to True")

        await sio.emit('camera_status', {'status': 'started', 'message': 'Camera started successfully'}, to=sid)
        print(f"[DEBUG] Sent camera_status event to client")

        # Start streaming metrics in background
        print(f"[DEBUG] Creating background task for streaming...")
        asyncio.create_task(stream_camera_metrics(sid))
        print(f"[DEBUG] Background task created")

    except Exception as e:
        print(f"[ERROR] Error starting camera: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('error', {'message': f'Failed to start camera: {str(e)}'}, to=sid)

@sio.event
async def start_interview(sid, data):
    """Start interview session"""
    global camera_active
    if not camera_active:
        await start_camera(sid, data)
    print(f"Client {sid} requested to start interview")
    
	# TODO: Initialize interview session from file that Vinh is working on

    await sio.emit('interview_status', {'status': 'started', 'message': 'Interview started'}, to=sid)

@sio.event
async def stop_camera(sid, data):
    """Stop camera processing"""
    global camera_active, video_capture

    print(f"Client {sid} requested to stop camera")
    camera_active = False

    if video_capture is not None:
        video_capture.release()
        video_capture = None

    await sio.emit('camera_status', {'status': 'stopped', 'message': 'Camera stopped'}, to=sid)


@sio.event
async def process_audio(sid, data):
    """Process audio data from frontend"""
    audio_data = data.get('audio', [])
    print(f"✓ Received audio data from {sid} - Length: {len(audio_data)} bytes")

    # TODO: Implement audio processing with voice.py
    # For now, acknowledge receipt and send back processing result

    result = {
        'status': 'success',
        'message': 'Audio received and processed successfully',
        'data': {
            'audio_length': len(audio_data),
            'timestamp': data.get('timestamp', None),
            'processed': True,
            # Add more processing results here when voice.py is integrated
        }
    }

    print(f"✓ Sending audio_processed acknowledgment to {sid}")
    await sio.emit('audio_processed', result, to=sid)


@sio.event
async def user_message(sid, data):
    """Receive user input/messages from frontend"""
    message = data.get('message', '')
    print(f"Received message from {sid}: {message}")

    # TODO: Send to AI agent for processing
    # For now, echo back
    await sio.emit('ai_response', {
        'message': f'Echo: {message}',
        'timestamp': data.get('timestamp')
    }, to=sid)


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def stream_camera_metrics(sid):
    """Stream camera metrics to connected client"""
    global camera_active, video_capture, posture_detector, current_frame, current_metrics

    print(f"[STREAM] Starting camera metrics stream for {sid}")
    frame_count = 0

    while camera_active:
        try:
            if video_capture is None or not video_capture.isOpened():
                print(f"[STREAM] Video capture not ready, waiting...")
                await asyncio.sleep(0.1)
                continue

            success, frame = video_capture.read()

            if not success:
                print(f"[STREAM] Failed to grab frame")
                await asyncio.sleep(0.1)
                continue

            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)

            # Process frame with posture detector
            annotated_frame, metrics = posture_detector.process_frame(frame)

            # Store current frame and metrics globally
            current_frame = annotated_frame
            current_metrics = metrics

            # Encode frame as base64 JPEG for transmission
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Prepare metrics data
            metrics_data = {}
            if metrics:
                metrics_data = {
                    'shoulder_angle': metrics.shoulder_angle,
                    'head_tilt': metrics.head_tilt,
                    'forward_lean': metrics.forward_lean,
                    'head_motion_score': metrics.head_motion_score,
                    'hand_motion_score': metrics.hand_motion_score,
                    'eye_contact_maintained': metrics.eye_contact_maintained,
                    'eye_contact_duration': metrics.eye_contact_duration,
                    'is_slouching': metrics.is_slouching,
                    'is_tilted': metrics.is_tilted,
                    'is_leaning': metrics.is_leaning,
                    'is_head_moving': metrics.is_head_moving,
                    'is_hand_fidgeting': metrics.is_hand_fidgeting,
                    'issues': metrics.issues,
                    'timestamp': metrics.timestamp
                }

            # Send both video frame and metrics to client
            await sio.emit('video_frame', {
                'frame': f'data:image/jpeg;base64,{frame_base64}',
                'metrics': metrics_data
            }, to=sid)

            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames (~1 second)
                print(f"[STREAM] Sent {frame_count} frames to {sid}")

            # Small delay to control frame rate (~30 FPS)
            await asyncio.sleep(0.033)

        except Exception as e:
            print(f"[STREAM ERROR] Error in stream_camera_metrics: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(0.1)

    print(f"[STREAM] Camera metrics stream stopped for {sid} (sent {frame_count} frames total)")


# ============================================================================
# HTTP ENDPOINTS (Alternative to Socket.IO for simple requests)
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Video Meeting Analyzer API",
        "endpoints": {
            "socket.io": "/socket.io",
            "video_feed": "/video_feed",
            "health": "/"
        }
    }


@app.get("/video_feed")
async def video_feed():
    """
    HTTP endpoint for streaming video (alternative to Socket.IO)
    Returns MJPEG stream
    """
    def generate_frames():
        global current_frame
        while True:
            if current_frame is not None:
                _, buffer = cv2.imencode('.jpg', current_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Send empty frame if no camera data
                import time
                time.sleep(0.1)

    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.get("/metrics")
async def get_metrics():
    """Get current metrics via HTTP"""
    global current_metrics

    if current_metrics is None:
        return {"status": "no_data", "metrics": None}

    return {
        "status": "ok",
        "metrics": {
            'shoulder_angle': current_metrics.shoulder_angle,
            'head_tilt': current_metrics.head_tilt,
            'forward_lean': current_metrics.forward_lean,
            'head_motion_score': current_metrics.head_motion_score,
            'hand_motion_score': current_metrics.hand_motion_score,
            'eye_contact_maintained': current_metrics.eye_contact_maintained,
            'issues': current_metrics.issues,
        }
    }


# ============================================================================
# CLEANUP
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global video_capture, posture_detector

    if video_capture is not None:
        video_capture.release()

    if posture_detector is not None:
        posture_detector.release()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("="*70)
    print("=� Starting Video Meeting Analyzer API")
    print("="*70)
    print("Socket.IO endpoint: http://localhost:8000/socket.io")
    print("HTTP video feed:    http://localhost:8000/video_feed")
    print("Health check:       http://localhost:8000/")
    print("="*70)

    uvicorn.run(
        "api:socket_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

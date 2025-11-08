import pyaudio
import numpy as np
import whisper
import wave
import os
import time
from dataclasses import dataclass
from typing import List, Optional
from collections import deque
import threading
import queue

@dataclass
class SpeechMetrics:
    """Data class to store speech analysis metrics"""
    text: str
    words_per_minute: float
    volume_db: float
    clarity_score: float  # Based on transcription confidence
    speech_duration: float
    pause_ratio: float  # Ratio of silence to speech
    is_too_fast: bool
    is_too_slow: bool
    is_too_quiet: bool
    is_unclear: bool
    timestamp: float
    issues: List[str]

class VoiceAnalyzer:
    """Analyzes speech patterns using Whisper and audio processing"""
    
    def __init__(self, model_size="base"):
        """
        Initialize the voice analyzer
        model_size: 'tiny', 'base', 'small', 'medium', 'large'
        'base' is a good balance of speed and accuracy for live analysis
        """
        print("Loading Whisper model...")
        self.model = whisper.load_model(model_size)
        print("Model loaded!")
        
        # Audio configuration
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # Whisper expects 16kHz
        self.RECORD_SECONDS = 5  # 5-second chunks
        
        # Analysis thresholds
        self.MIN_WPM = 120  # Words per minute - too slow
        self.MAX_WPM = 180  # Words per minute - too fast
        self.IDEAL_WPM_RANGE = (130, 160)  # Ideal speaking rate
        self.MIN_VOLUME_DB = -60  # Minimum volume in dB
        self.MIN_CLARITY_SCORE = 0.7  # Minimum clarity (based on word confidence)
        self.MAX_PAUSE_RATIO = 0.4  # Maximum ratio of silence
        
        # Audio interface
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # For tracking metrics over session
        self.session_metrics = []
        
        # Threading for non-blocking analysis
        self.analysis_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_analyzing = False
        
    def calculate_volume_db(self, audio_data):
        """Calculate volume in decibels"""
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate RMS (Root Mean Square)
        rms = np.sqrt(np.mean(audio_array**2))
        
        # Convert to dB (with reference to max int16 value)
        if rms > 0:
            db = 20 * np.log10(rms / 32768.0)
        else:
            db = -100  # Very quiet
        
        return db
    
    def detect_speech_segments(self, audio_data):
        """
        Detect speech vs silence segments using adaptive thresholding
        Returns ratio of silence to total duration
        """
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate energy in chunks
        chunk_size = self.RATE // 10  # 100ms chunks
        num_chunks = len(audio_array) // chunk_size
        
        if num_chunks == 0:
            return 0.0
        
        # Calculate RMS for each chunk
        chunk_energies = []
        for i in range(num_chunks):
            chunk = audio_array[i * chunk_size:(i + 1) * chunk_size]
            rms = np.sqrt(np.mean(chunk**2))
            chunk_energies.append(rms)
        
        # Use adaptive threshold: mean energy * 0.2
        # This adjusts based on the actual volume of the recording
        mean_energy = np.mean(chunk_energies)
        threshold_energy = mean_energy * 0.2
        
        # Count chunks above threshold as speech
        speech_chunks = sum(1 for energy in chunk_energies if energy > threshold_energy)
        
        silence_ratio = 1.0 - (speech_chunks / num_chunks)
        return silence_ratio
    
    def calculate_speech_rate(self, text: str, duration: float) -> float:
        """Calculate words per minute"""
        if duration == 0:
            return 0.0
        
        words = text.split()
        word_count = len(words)
        
        # Convert to words per minute
        wpm = (word_count / duration) * 60
        return wpm
    
    def estimate_clarity(self, result) -> float:
        """
        Estimate clarity based on Whisper's internal metrics
        Uses segment-level log probabilities if available
        """
        try:
            # Whisper provides segment-level confidence
            if hasattr(result, 'segments') and result.segments:
                # Average the confidence across segments
                avg_logprob = np.mean([seg['avg_logprob'] for seg in result.segments])
                # Convert log probability to a 0-1 score
                # Typical range is -1.0 to 0.0, with closer to 0 being better
                clarity = np.exp(avg_logprob)
                return min(1.0, max(0.0, clarity))
            else:
                # Fallback: assume decent clarity if transcribed
                return 0.8 if result['text'].strip() else 0.0
        except:
            return 0.8  # Default moderate clarity
    
    def analyze_audio_chunk(self, audio_data, duration: float) -> Optional[SpeechMetrics]:
        """Analyze a chunk of audio data"""
        
        # Save audio to temporary file for Whisper
        temp_filename = "temp_audio.wav"
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(audio_data)
        
        # Transcribe with Whisper
        try:
            result = self.model.transcribe(
                temp_filename,
                language="en",
                fp16=False,
                verbose=False
            )
            
            text = result['text'].strip()
            
            # Skip if no speech detected
            if not text or len(text) < 3:
                os.remove(temp_filename)
                return None
            
            # Calculate metrics
            volume_db = self.calculate_volume_db(audio_data)
            pause_ratio = self.detect_speech_segments(audio_data)
            wpm = self.calculate_speech_rate(text, duration)
            clarity = self.estimate_clarity(result)
            
            # Detect issues
            issues = []
            is_too_fast = wpm > self.MAX_WPM
            is_too_slow = wpm < self.MIN_WPM and wpm > 0
            is_too_quiet = volume_db < self.MIN_VOLUME_DB
            is_unclear = clarity < self.MIN_CLARITY_SCORE
            
            if is_too_fast:
                issues.append(f"Speaking too fast ({wpm:.0f} WPM)")
            elif is_too_slow:
                issues.append(f"Speaking too slow ({wpm:.0f} WPM)")
            
            if wpm > 0 and self.IDEAL_WPM_RANGE[0] <= wpm <= self.IDEAL_WPM_RANGE[1]:
                issues.append(f"Good pace ({wpm:.0f} WPM)")
            
            if is_too_quiet:
                issues.append(f"Volume too low ({volume_db:.1f} dB)")
            
            if is_unclear:
                issues.append(f"Unclear speech (clarity: {clarity:.2f})")
            
            if pause_ratio > self.MAX_PAUSE_RATIO:
                issues.append(f"Too many pauses ({pause_ratio*100:.0f}% silence)")
            
            metrics = SpeechMetrics(
                text=text,
                words_per_minute=wpm,
                volume_db=volume_db,
                clarity_score=clarity,
                speech_duration=duration,
                pause_ratio=pause_ratio,
                is_too_fast=is_too_fast,
                is_too_slow=is_too_slow,
                is_too_quiet=is_too_quiet,
                is_unclear=is_unclear,
                timestamp=time.time(),
                issues=issues if issues else ["Good speech"]
            )
            
            # Clean up temp file
            os.remove(temp_filename)
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            return None
    
    def analysis_worker(self):
        """Worker thread for analyzing audio chunks"""
        while self.is_analyzing:
            try:
                audio_data, duration = self.analysis_queue.get(timeout=0.1)
                metrics = self.analyze_audio_chunk(audio_data, duration)
                if metrics:
                    self.result_queue.put(metrics)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {e}")
    
    def start_listening(self):
        """Start listening to microphone"""
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # Start analysis thread
        self.is_analyzing = True
        self.analysis_thread = threading.Thread(target=self.analysis_worker)
        self.analysis_thread.start()
        
        print("Listening... Speak into your microphone!")
        print(f"Analyzing in {self.RECORD_SECONDS}-second chunks")
        print("-" * 60)
    
    def record_chunk(self) -> bytes:
        """Record a chunk of audio"""
        frames = []
        
        for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        return b''.join(frames)
    
    def stop_listening(self):
        """Stop listening and clean up"""
        self.is_analyzing = False
        
        if self.analysis_thread:
            self.analysis_thread.join()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
    
    def print_metrics(self, metrics: SpeechMetrics):
        """Print formatted metrics"""
        print(f"\n{'='*60}")
        print(f"Transcription: {metrics.text}")
        print(f"{'-'*60}")
        print(f"Speaking Rate: {metrics.words_per_minute:.1f} WPM", end="")
        
        if self.IDEAL_WPM_RANGE[0] <= metrics.words_per_minute <= self.IDEAL_WPM_RANGE[1]:
            print(" ✓ (Good pace)")
        elif metrics.is_too_fast:
            print(" ⚠ (Too fast - slow down)")
        elif metrics.is_too_slow:
            print(" ⚠ (Too slow - speed up)")
        else:
            print()
        
        print(f"Volume: {metrics.volume_db:.1f} dB", end="")
        if metrics.is_too_quiet:
            print(f" ⚠ (Speak louder - need >{self.MIN_VOLUME_DB} dB)")
        else:
            print(" ✓")
        
        print(f"Clarity: {metrics.clarity_score:.2f}", end="")
        if metrics.is_unclear:
            print(" ⚠ (Enunciate more clearly)")
        else:
            print(" ✓")
        
        print(f"Pause Ratio: {metrics.pause_ratio*100:.1f}%", end="")
        if metrics.pause_ratio > self.MAX_PAUSE_RATIO:
            print(" ⚠ (Reduce pauses)")
        else:
            print(" ✓")
        
        print(f"\nFeedback: {', '.join(metrics.issues)}")
        print(f"{'='*60}\n")


def main():
    """Main function to run the voice analyzer"""
    
    print("Initializing Speech Analyzer...")
    analyzer = VoiceAnalyzer(model_size="base")
    
    print("\nInstructions:")
    print("- Speak naturally into your microphone")
    print("- Analysis happens every 5 seconds")
    print("- Press Ctrl+C to stop")
    print("\nOptimal speaking rate: 130-160 words per minute")
    print("Ideal volume: Above -60 dB")
    print("\nStarting in 3 seconds...\n")
    
    time.sleep(3)
    
    try:
        analyzer.start_listening()
        
        chunk_count = 0
        while True:
            chunk_count += 1
            print(f"Recording chunk {chunk_count}... (5 seconds)")
            
            # Record audio chunk
            audio_data = analyzer.record_chunk()
            
            # Add to analysis queue
            analyzer.analysis_queue.put((audio_data, analyzer.RECORD_SECONDS))
            
            # Check for results
            try:
                while True:
                    metrics = analyzer.result_queue.get_nowait()
                    analyzer.session_metrics.append(metrics)
                    analyzer.print_metrics(metrics)
            except queue.Empty:
                pass
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        analyzer.stop_listening()
        
        # Print session summary
        if analyzer.session_metrics:
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            
            total_chunks = len(analyzer.session_metrics)
            avg_wpm = np.mean([m.words_per_minute for m in analyzer.session_metrics if m.words_per_minute > 0])
            avg_volume = np.mean([m.volume_db for m in analyzer.session_metrics])
            avg_clarity = np.mean([m.clarity_score for m in analyzer.session_metrics])
            
            too_fast = sum(1 for m in analyzer.session_metrics if m.is_too_fast)
            too_slow = sum(1 for m in analyzer.session_metrics if m.is_too_slow)
            too_quiet = sum(1 for m in analyzer.session_metrics if m.is_too_quiet)
            unclear = sum(1 for m in analyzer.session_metrics if m.is_unclear)
            
            print(f"\nTotal Chunks Analyzed: {total_chunks}")
            print(f"\nAverage Metrics:")
            print(f"  Speaking Rate: {avg_wpm:.1f} WPM")
            print(f"  Volume: {avg_volume:.1f} dB")
            print(f"  Clarity: {avg_clarity:.2f}")
            
            print(f"\nIssue Breakdown:")
            print(f"  Too Fast: {too_fast} chunks ({too_fast/total_chunks*100:.1f}%)")
            print(f"  Too Slow: {too_slow} chunks ({too_slow/total_chunks*100:.1f}%)")
            print(f"  Too Quiet: {too_quiet} chunks ({too_quiet/total_chunks*100:.1f}%)")
            print(f"  Unclear: {unclear} chunks ({unclear/total_chunks*100:.1f}%)")
            
            print("\n" + "="*60)


if __name__ == "__main__":
    main()
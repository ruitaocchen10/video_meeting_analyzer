"use client";

import { useEffect, useState, useRef } from "react";
import { io, Socket } from "socket.io-client";
import Node from "../components/Node";

interface VideoFrame {
  frame: string;  // base64 encoded image
}

export default function HomePage() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [videoFrame, setVideoFrame] = useState<string | null>(null);
  const [aiResponse, setAiResponse] = useState<string>("Waiting for AI response...");
  const [audioStatus, setAudioStatus] = useState<string>("No audio processed yet");

  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    // Initialize Socket.IO connection when component mounts
    const newSocket = io("http://localhost:8000", {
      transports: ["websocket", "polling"],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socketRef.current = newSocket;
    setSocket(newSocket);

    // Connection event handlers
    newSocket.on("connect", () => {
      console.log("Connected to backend!");
      setConnected(true);
    });

    newSocket.on("disconnect", () => {
      console.log("Disconnected from backend");
      setConnected(false);
      setCameraActive(false);
      setInterviewStarted(false);
    });

    newSocket.on("connection_status", (data) => {
      console.log("Connection status:", data);
    });

    // Camera event handlers
    newSocket.on("camera_status", (data) => {
      console.log("Camera status:", data);
      if (data.status === "started") {
        setCameraActive(true);
      } else if (data.status === "stopped") {
        setCameraActive(false);
      }
    });

    // Interview event handler
    newSocket.on("interview_status", (data) => {
      console.log("Interview status:", data);
      if (data.status === "started") {
        setInterviewStarted(true);
      } else if (data.status === "stopped") {
        setInterviewStarted(false);
      }
    });

    // Video frame handler - receives processed video
    newSocket.on("video_frame", (data: VideoFrame) => {
      setVideoFrame(data.frame);
    });

    // AI response handler
    newSocket.on("ai_response", (data) => {
      console.log("AI response:", data);
      setAiResponse(data.message);
    });

    // Audio processed handler
    newSocket.on("audio_processed", (data) => {
      console.log("Audio processed:", data);
      if (data.status === "success") {
        const audioLength = data.data?.audio_length || 0;
        setAudioStatus(`Audio processed successfully! Length: ${audioLength} bytes`);
      } else {
        setAudioStatus(`Audio processing status: ${data.status}`);
      }
    });

    // Error handler
    newSocket.on("error", (data) => {
      console.error("Error from backend:", data);
      alert(`Error: ${data.message}`);
    });

    // Cleanup on unmount
    return () => {
      newSocket.close();
    };
  }, []);

  const startCamera = () => {
    if (socket && connected) {
      socket.emit("start_camera", {});
    }
  };

  const stopCamera = () => {
    if (socket && connected) {
      socket.emit("stop_camera", {});
    }
  };

  const startInterview = () => {
    if (socket && connected) {
      socket.emit("start_interview", {});
    }
  };

  const handleReadyClick = () => {
    // TODO: Implement ready button functionality
  };

      // socket.emit("user_message", {
      //   message: "Hello from frontend!",
      //   timestamp: Date.now(),
      // });



  return (
    <div className="flex flex-col w-[90vw] h-[90vh] bg-[#1e1e1e] rounded-[10px] shadow-[0_0_px_rgba(0,0,0,0.2)] p-5 bg-gradient-to-br from-gray-900 via-gray-800 to-slate-900">
      {/* Connection Status Bar */}
      <div className="mb-4 p-3 rounded-lg bg-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`} />
          <span className="text-sm text-gray-300">
            {connected ? "Connected to Backend" : "Disconnected"}
          </span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={startCamera}
            disabled={!connected || cameraActive}
            className="px-4 py-2 text-sm bg-emerald-500 hover:bg-emerald-600 disabled:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition"
          >
            {cameraActive ? "Camera Active" : "Start Camera"}
          </button>
          <button
            onClick={stopCamera}
            disabled={!connected || !cameraActive}
            className="px-4 py-2 text-sm bg-red-500 hover:bg-red-600 disabled:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition"
          >
            Stop Camera
          </button>
          <button
            onClick={startInterview}
            disabled={!connected || interviewStarted}
            className="px-4 py-2 text-sm bg-blue-500 hover:bg-blue-600 disabled:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition"
          >
            Start Interview
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-row flex-1 gap-5">
        {/* Video Panel */}
        <div className="flex-[3] flex flex-col justify-center items-center relative">
          <h1 className="font-light m-0 mb-5 tracking-[2px]">
            Live Video Stream
          </h1>

          {videoFrame ? (
            <img
              src={videoFrame}
              alt="Live Stream"
              className="max-h-[85%] rounded-lg object-cover transition-transform duration-300"
            />
          ) : (
            <div className="max-h-[85%] w-full rounded-lg bg-gray-800 flex items-center justify-center text-gray-500">
              {cameraActive ? "Waiting for video..." : "Camera not active"}
            </div>
          )}
        </div>

        {/* Data Panel */}
        <div className="relative flex-1 flex justify-center items-center border-l border-[#444] pl-5">
          <div className="w-full h-full flex flex-col gap-3 overflow-y-auto pr-1.5 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-[#1e1e1e] [&::-webkit-scrollbar-track]:rounded [&::-webkit-scrollbar-thumb]:bg-[#555] [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb:hover]:bg-[#666]">
            <Node
              title="Interview Questions"
              content={aiResponse}
            />
            <Node
              title="Speech Analysis"
              content={audioStatus}
            />
            <Node
              title="Practice Summary"
              content="Speech pacing, clarity, and volume metrics will appear here."
            />
          </div>

          <button
            onClick={handleReadyClick}
            disabled={!connected || !interviewStarted}
            className="absolute bottom-4 left-4 w-16 h-16 rounded-full bg-blue-500 text-white font-semibold text-sm disabled:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-transform duration-150 hover:scale-110 active:scale-95 enabled:hover:bg-blue-600"
          >
            Ready!
          </button>
        </div>
      </div>
    </div>
  );
}

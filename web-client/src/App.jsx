import React, { useState } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import {
  Upload, Box, Download, Share2, Maximize2, RotateCw, X, Layers, Github,
  ArrowUp, ArrowDown, ArrowLeft, ArrowRight, RotateCcw, Monitor, Target, Hand,
  FileJson, MapPin, CheckCircle, AlertCircle, Image as ImageIcon
} from 'lucide-react';
import * as THREE from 'three';
import axios from 'axios';
import { Canvas } from '@react-three/fiber';
import { Splat, OrbitControls, Loader } from '@react-three/drei';

/**
 * SHARP Web Interface
 * * A modern, fluid frontend for Apple's SHARP (Sharp Monocular View Synthesis) model.
 */

const SharpApp = () => {
  const [appState, setAppState] = useState('idle'); // idle, uploading, processing, complete
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isHovering, setIsHovering] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [cameraTarget, setCameraTarget] = useState(null); // { position: [x,y,z], target: [0,0,0] }
  const [isRecentering, setIsRecentering] = useState(false);
  const [interactionMode, setInteractionMode] = useState('rotate'); // rotate, pan
  const [is360, setIs360] = useState(false);
  const [uploadMode, setUploadMode] = useState('single'); // 'single' | 'multistation'
  const [stationsFile, setStationsFile] = useState(null);
  const [stationsData, setStationsData] = useState(null); // parsed JSON
  const [stationImages, setStationImages] = useState([]); // array of File objects
  const [matchStatus, setMatchStatus] = useState(null); // { matched: [], unmatched: [] }
  const [isMultistation, setIsMultistation] = useState(false);

  // Constants
  const API_BASE = 'http://localhost:8000';

  // Check backend health
  const checkBackendHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE}/`, { timeout: 3000 });
      return response.data.status === 'running';
    } catch (error) {
      console.error("Backend health check failed:", error);
      return false;
    }
  };

  // Detect if an image is a 360 equirectangular panorama (aspect ratio ~2:1)
  const detect360Image = (file) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const ratio = img.width / img.height;
        const is360 = ratio >= 1.9 && ratio <= 2.1;
        URL.revokeObjectURL(img.src);
        resolve(is360);
      };
      img.onerror = () => {
        URL.revokeObjectURL(img.src);
        resolve(false);
      };
      img.src = URL.createObjectURL(file);
    });
  };

  // --- Multi-Station Handlers ---

  const handleStationsJsonUpload = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (!data.stations || !Array.isArray(data.stations)) {
          alert("Invalid stations.json: missing 'stations' array");
          return;
        }
        setStationsFile(file);
        setStationsData(data);
        // Re-run matching if images are already loaded
        if (stationImages.length > 0) {
          matchFiles(data.stations, stationImages);
        }
      } catch (err) {
        alert("Failed to parse JSON file: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  const handleStationImagesUpload = (files) => {
    const imageFiles = Array.from(files).filter(f =>
      f.type.startsWith('image/') || f.name.toLowerCase().endsWith('.jpg') || f.name.toLowerCase().endsWith('.jpeg') || f.name.toLowerCase().endsWith('.png')
    );
    setStationImages(imageFiles);
    // Re-run matching if JSON is already loaded
    if (stationsData) {
      matchFiles(stationsData.stations, imageFiles);
    }
  };

  const matchFiles = (stations, images) => {
    const imageBasenames = images.map(f => f.name);
    const matched = [];
    const unmatched = [];

    for (const station of stations) {
      const expectedBasename = station.path_to_image.split('/').pop();
      const found = imageBasenames.includes(expectedBasename);
      if (found) {
        matched.push({ station, filename: expectedBasename });
      } else {
        unmatched.push({ station, expected: expectedBasename });
      }
    }

    setMatchStatus({ matched, unmatched });
  };

  const startMultistationProcessing = async () => {
    setAppState('processing');
    setProgress(0);
    setIsMultistation(true);
    setStatusMessage("Checking backend connection...");

    const isBackendHealthy = await checkBackendHealth();
    if (!isBackendHealthy) {
      setAppState('idle');
      setStatusMessage("");
      alert(
        "Backend server is not reachable.\n\n" +
        "Please ensure the backend is running:\n" +
        "1. Open a terminal\n" +
        "2. Navigate to the project root\n" +
        "3. Run: uvicorn backend.main:app --reload --port 8000"
      );
      return;
    }

    setStatusMessage(`Uploading ${stationImages.length} images + stations.json...`);

    const formData = new FormData();
    formData.append('stations_json', stationsFile);
    for (const img of stationImages) {
      formData.append('files', img);
    }

    try {
      const submitResponse = await axios.post(`${API_BASE}/predict-multistation`, formData, {
        timeout: 120000, // 2 minute timeout for multi-file upload
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const jobId = submitResponse.data.job_id;
      const stationsMatched = submitResponse.data.stations_matched;
      console.log(`Multi-station job submitted: ${jobId} (${stationsMatched} stations)`);

      // Poll with high tolerance for multi-station jobs
      const POLL_INTERVAL = 3000;
      const POLL_TIMEOUT = 30000;
      const MAX_POLL_ERRORS = 120; // Very high tolerance for long jobs
      let pollErrorCount = 0;
      let cancelled = false;

      const poll = async () => {
        if (cancelled) return;
        try {
          const statusResponse = await axios.get(`${API_BASE}/jobs/${jobId}`, {
            timeout: POLL_TIMEOUT,
          });
          const job = statusResponse.data;

          pollErrorCount = 0;
          setProgress(job.progress);
          setStatusMessage(job.message);

          if (job.status === 'complete') {
            setStatusMessage("Downloading result...");
            const resultResponse = await axios.get(`${API_BASE}/jobs/${jobId}/result`, {
              responseType: 'blob',
              timeout: 600000, // 10 minute timeout for very large multi-station splats
            });

            const splatUrl = window.URL.createObjectURL(new Blob([resultResponse.data]));
            setDownloadUrl(splatUrl);
            setAppState('complete');
            return;
          } else if (job.status === 'failed') {
            throw new Error(job.error || "Job failed");
          }
        } catch (pollError) {
          pollErrorCount++;
          console.error(`Polling error (${pollErrorCount}/${MAX_POLL_ERRORS}):`, pollError);

          if (pollError.message && pollError.message.includes("Job failed")) {
            setStatusMessage("Error: " + pollError.message);
            setAppState('idle');
            alert("Processing failed: " + pollError.message);
            return;
          } else if (pollErrorCount >= MAX_POLL_ERRORS) {
            setStatusMessage("Connection lost");
            setAppState('idle');
            alert(
              "Lost connection to backend server.\n\n" +
              "The backend may have stopped responding. Please check:\n" +
              "1. Is the backend still running?\n" +
              "2. Check the backend terminal for errors\n" +
              "3. Try refreshing the page and uploading again"
            );
            return;
          }
        }

        if (!cancelled) {
          setTimeout(poll, POLL_INTERVAL);
        }
      };

      setTimeout(poll, POLL_INTERVAL);

    } catch (error) {
      console.error("Error starting multi-station job:", error);
      setStatusMessage("");
      setAppState('idle');

      let errorMessage = "Failed to start multi-station processing.\n\n";
      if (error.response) {
        errorMessage += `Server error (${error.response.status}): ${error.response.data?.detail || error.response.statusText}`;
      } else if (error.request) {
        errorMessage += "Backend server did not respond.";
      } else {
        errorMessage += `Error: ${error.message}`;
      }
      alert(errorMessage);
    }
  };

  // --- Single Image Handlers ---

  const handleFileUpload = async (file) => {
    if (!file) return;

    // Detect 360 panorama before processing
    const detected360 = await detect360Image(file);
    setIs360(detected360);
    if (detected360) {
      console.log("360 panorama detected (aspect ratio ~2:1)");
    }

    // Standard preview logic
    const setPreview = (blob) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target.result);
        // Start processing with the ORIGINAL file
        startProcessing(file, detected360);
      };
      reader.readAsDataURL(blob);
    };

    // Check for HEIC/HEIF
    const isHeic = file.type === 'image/heic' ||
      file.type === 'image/heif' ||
      file.name.toLowerCase().endsWith('.heic') ||
      file.name.toLowerCase().endsWith('.heif');

    if (isHeic) {
      // Show immediate feedback
      setAppState('uploading');
      setStatusMessage("Converting HEIC for preview...");

      try {
        const heic2any = (await import('heic2any')).default;
        const convertedBlob = await heic2any({
          blob: file,
          toType: "image/jpeg",
          quality: 0.8
        });

        const jpgBlob = Array.isArray(convertedBlob) ? convertedBlob[0] : convertedBlob;
        setPreview(jpgBlob);
      } catch (e) {
        console.error("HEIC conversion failed:", e);
        // Fallback: try to just show it (won't work in Chrome but ok in Safari maybe?) 
        // Or just proceed without preview
        setStatusMessage("Preview failed, but processing...");
        // Still proceed to backend
        setUploadedImage(null); // No broken image
        startProcessing(file, detected360);
      }
    } else {
      // Normal image
      setAppState('uploading');
      setPreview(file);
    }
  };

  const startProcessing = async (file, use360 = false) => {
    setAppState('processing');
    setProgress(0);
    setStatusMessage("Checking backend connection...");

    // First, check if backend is reachable
    const isBackendHealthy = await checkBackendHealth();
    if (!isBackendHealthy) {
      setAppState('idle');
      setStatusMessage("");
      alert(
        "Backend server is not reachable.\n\n" +
        "Please ensure the backend is running:\n" +
        "1. Open a terminal\n" +
        "2. Navigate to the project root\n" +
        "3. Run: uvicorn backend.main:app --reload --port 8000\n\n" +
        "The backend should be running at http://localhost:8000"
      );
      return;
    }

    // Choose endpoint based on 360 detection
    const endpoint = use360 ? `${API_BASE}/predict360` : `${API_BASE}/predict`;
    setStatusMessage(use360 ? "Uploading 360 panorama..." : "Initializing upload...");

    const formData = new FormData();
    formData.append('file', file);

    try {
      // 1. Submit Job
      setStatusMessage(use360 ? "Uploading 360 panorama..." : "Uploading image...");
      const submitResponse = await axios.post(endpoint, formData, {
        timeout: 30000, // 30 second timeout for upload
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      const jobId = submitResponse.data.job_id;

      console.log("Job submitted:", jobId);

      // 2. Poll Status (sequential - wait for each response before polling again)
      const POLL_INTERVAL = use360 ? 2000 : 1000; // 360 jobs take much longer
      const POLL_TIMEOUT = 30000; // 30s timeout per poll (heavy processing can slow responses)
      const MAX_POLL_ERRORS = use360 ? 60 : 30; // 360 jobs need more tolerance
      let pollErrorCount = 0;
      let cancelled = false;

      const poll = async () => {
        if (cancelled) return;
        try {
          const statusResponse = await axios.get(`${API_BASE}/jobs/${jobId}`, {
            timeout: POLL_TIMEOUT,
          });
          const job = statusResponse.data;

          pollErrorCount = 0; // Reset error count on success
          setProgress(job.progress);
          setStatusMessage(job.message);

          if (job.status === 'complete') {
            // 3. Get Result
            setStatusMessage("Downloading result...");
            const resultResponse = await axios.get(`${API_BASE}/jobs/${jobId}/result`, {
              responseType: 'blob',
              timeout: 300000, // 5 minute timeout for large 360 splat downloads
            });

            const splatUrl = window.URL.createObjectURL(new Blob([resultResponse.data]));
            setDownloadUrl(splatUrl);
            setAppState('complete');
            return; // Done - don't schedule another poll
          } else if (job.status === 'failed') {
            throw new Error(job.error || "Job failed");
          }
        } catch (pollError) {
          pollErrorCount++;
          console.error(`Polling error (${pollErrorCount}/${MAX_POLL_ERRORS}):`, pollError);

          if (pollError.message && pollError.message.includes("Job failed")) {
            setStatusMessage("Error: " + pollError.message);
            setAppState('idle');
            alert("Processing failed: " + pollError.message);
            return; // Done - don't schedule another poll
          } else if (pollErrorCount >= MAX_POLL_ERRORS) {
            // Too many consecutive errors, likely backend issue
            setStatusMessage("Connection lost");
            setAppState('idle');
            alert(
              "Lost connection to backend server.\n\n" +
              "The backend may have stopped responding. Please check:\n" +
              "1. Is the backend still running?\n" +
              "2. Check the backend terminal for errors\n" +
              "3. Try refreshing the page and uploading again"
            );
            return; // Done - don't schedule another poll
          }
          // Otherwise, keep polling (transient error)
        }

        // Schedule next poll (only after current one completes)
        if (!cancelled) {
          setTimeout(poll, POLL_INTERVAL);
        }
      };

      // Start polling
      setTimeout(poll, POLL_INTERVAL);

    } catch (error) {
      console.error("Error starting job:", error);
      setStatusMessage("");
      setAppState('idle');
      
      let errorMessage = "Failed to start processing.\n\n";
      
      if (error.code === 'ECONNREFUSED' || error.message.includes('Network Error')) {
        errorMessage += "Cannot connect to backend server.\n\n";
        errorMessage += "Please ensure the backend is running:\n";
        errorMessage += "1. Open a terminal\n";
        errorMessage += "2. Navigate to the project root\n";
        errorMessage += "3. Run: uvicorn backend.main:app --reload --port 8000\n\n";
        errorMessage += "The backend should be running at http://localhost:8000";
      } else if (error.response) {
        // Server responded with error status
        errorMessage += `Server error (${error.response.status}): ${error.response.data?.detail || error.response.statusText}`;
      } else if (error.request) {
        // Request made but no response
        errorMessage += "Backend server did not respond. Please check if it's running.";
      } else {
        errorMessage += `Error: ${error.message}`;
      }
      
      alert(errorMessage);
    }
  };

  const resetApp = () => {
    setAppState('idle');
    setUploadedImage(null);
    setProgress(0);
    setDownloadUrl(null);
    setCameraTarget(null);
    setIsRecentering(false);
    setInteractionMode('rotate');
    setIs360(false);
    setIsMultistation(false);
    setStationsFile(null);
    setStationsData(null);
    setStationImages([]);
    setMatchStatus(null);
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-white font-sans selection:bg-blue-500/30 overflow-hidden relative">

      {/* Background Gradients */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-blue-900/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-indigo-900/10 rounded-full blur-[120px] pointer-events-none" />

      {/* Navigation */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Box size={18} className="text-white" />
          </div>
          <span className="text-xl font-medium tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-neutral-400">
            SHARP <span className="font-light text-neutral-500">Web</span>
          </span>
        </div>
        <div className="flex items-center gap-6 text-sm font-medium text-neutral-400">
          <a
            href="https://github.com/apple/ml-sharp"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-white transition-colors flex items-center gap-2"
          >
            <Github size={16} />
            Apple Documentation
          </a>
          <a
            href="https://superspl.at/editor"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/10 rounded-full text-white transition-all backdrop-blur-sm"
          >
            Open SuperSplat Editor
          </a>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="relative z-10 max-w-5xl mx-auto px-6 h-[calc(100vh-100px)] flex flex-col items-center justify-center">

        {/* State: Idle / Drop Zone */}
        {appState === 'idle' && (
          <div className="w-full max-w-3xl flex flex-col items-center gap-6">
            {/* Mode Toggle */}
            <div className="flex items-center gap-1 p-1 bg-neutral-900/60 rounded-full border border-white/5 backdrop-blur-sm">
              <button
                onClick={() => setUploadMode('single')}
                className={`px-5 py-2 rounded-full text-sm font-medium transition-all ${
                  uploadMode === 'single'
                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/30'
                    : 'text-neutral-400 hover:text-white'
                }`}
              >
                Single Image / 360
              </button>
              <button
                onClick={() => setUploadMode('multistation')}
                className={`px-5 py-2 rounded-full text-sm font-medium transition-all flex items-center gap-2 ${
                  uploadMode === 'multistation'
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/30'
                    : 'text-neutral-400 hover:text-white'
                }`}
              >
                <MapPin size={14} />
                Multi-Station Capture
              </button>
            </div>

            {/* Single Image Upload */}
            {uploadMode === 'single' && (
              <div
                className={`
                  w-full max-w-2xl aspect-[16/9] rounded-3xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center gap-6 group cursor-pointer relative overflow-hidden
                  ${isHovering ? 'border-blue-500 bg-blue-500/5' : 'border-neutral-800 bg-neutral-900/30 hover:bg-neutral-900/50 hover:border-neutral-700'}
                `}
                onDragOver={(e) => { e.preventDefault(); setIsHovering(true); }}
                onDragLeave={() => setIsHovering(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsHovering(false);
                  const file = e.dataTransfer.files[0];
                  if (file && file.type.startsWith('image/')) handleFileUpload(file);
                }}
              >
                <input
                  type="file"
                  className="absolute inset-0 opacity-0 cursor-pointer z-20"
                  onChange={(e) => handleFileUpload(e.target.files[0])}
                  accept="image/*,.heic,.HEIC,.heif,.HEIF"
                />

                <div className="p-4 bg-neutral-800/50 rounded-2xl group-hover:scale-110 transition-transform duration-300 shadow-xl border border-white/5">
                  <Upload size={32} className="text-blue-400" />
                </div>

                <div className="text-center space-y-2 relative z-10">
                  <h2 className="text-2xl font-light text-white">
                    Drag and drop your image
                  </h2>
                  <p className="text-neutral-500 max-w-xs mx-auto">
                    Upload a single photo or 360 panorama to generate a photorealistic 3D Gaussian Splat scene.
                  </p>
                </div>

                <div className="absolute bottom-8 flex gap-8 text-neutral-700 font-mono text-xs uppercase tracking-widest opacity-50">
                  <span>Single Shot</span>
                  <span>.</span>
                  <span>360 Panorama</span>
                  <span>.</span>
                  <span>.SPLAT Output</span>
                </div>
              </div>
            )}

            {/* Multi-Station Upload */}
            {uploadMode === 'multistation' && (
              <div className="w-full flex flex-col gap-6">
                <div className="grid grid-cols-2 gap-4">
                  {/* JSON Upload Area */}
                  <div className="relative rounded-2xl border-2 border-dashed border-neutral-800 bg-neutral-900/30 hover:bg-neutral-900/50 hover:border-neutral-700 transition-all p-6 flex flex-col items-center justify-center gap-4 min-h-[200px]">
                    <input
                      type="file"
                      className="absolute inset-0 opacity-0 cursor-pointer z-10"
                      accept=".json"
                      onChange={(e) => handleStationsJsonUpload(e.target.files[0])}
                    />
                    {stationsFile ? (
                      <>
                        <div className="p-3 bg-green-500/10 rounded-xl border border-green-500/20">
                          <CheckCircle size={24} className="text-green-400" />
                        </div>
                        <div className="text-center space-y-1">
                          <p className="text-sm font-medium text-white">{stationsFile.name}</p>
                          <p className="text-xs text-green-400">
                            {stationsData?.stations?.length || 0} stations found
                          </p>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="p-3 bg-neutral-800/50 rounded-xl border border-white/5">
                          <FileJson size={24} className="text-indigo-400" />
                        </div>
                        <div className="text-center space-y-1">
                          <p className="text-sm font-medium text-white">Upload stations.json</p>
                          <p className="text-xs text-neutral-500">Station positions and orientations</p>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Images Upload Area */}
                  <div className="relative rounded-2xl border-2 border-dashed border-neutral-800 bg-neutral-900/30 hover:bg-neutral-900/50 hover:border-neutral-700 transition-all p-6 flex flex-col items-center justify-center gap-4 min-h-[200px]">
                    <input
                      type="file"
                      className="absolute inset-0 opacity-0 cursor-pointer z-10"
                      accept="image/*"
                      multiple
                      onChange={(e) => handleStationImagesUpload(e.target.files)}
                    />
                    {stationImages.length > 0 ? (
                      <>
                        <div className="p-3 bg-green-500/10 rounded-xl border border-green-500/20">
                          <ImageIcon size={24} className="text-green-400" />
                        </div>
                        <div className="text-center space-y-1">
                          <p className="text-sm font-medium text-white">{stationImages.length} images selected</p>
                          <p className="text-xs text-neutral-400">Click to change selection</p>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="p-3 bg-neutral-800/50 rounded-xl border border-white/5">
                          <Upload size={24} className="text-indigo-400" />
                        </div>
                        <div className="text-center space-y-1">
                          <p className="text-sm font-medium text-white">Upload 360 Photos</p>
                          <p className="text-xs text-neutral-500">Select all station images</p>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Match Status */}
                {matchStatus && (
                  <div className="rounded-2xl bg-neutral-900/50 border border-white/5 p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-white">Station Matching</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        matchStatus.unmatched.length === 0
                          ? 'bg-green-500/20 text-green-400 border border-green-500/20'
                          : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/20'
                      }`}>
                        {matchStatus.matched.length}/{matchStatus.matched.length + matchStatus.unmatched.length} matched
                      </span>
                    </div>
                    <div className="max-h-[200px] overflow-y-auto space-y-1.5">
                      {matchStatus.matched.map((m, i) => (
                        <div key={`matched-${i}`} className="flex items-center gap-2 text-xs">
                          <CheckCircle size={12} className="text-green-400 shrink-0" />
                          <span className="text-neutral-300 truncate">{m.station.name || m.station.id}</span>
                          <span className="text-neutral-600 truncate ml-auto">{m.filename}</span>
                        </div>
                      ))}
                      {matchStatus.unmatched.map((u, i) => (
                        <div key={`unmatched-${i}`} className="flex items-center gap-2 text-xs">
                          <AlertCircle size={12} className="text-yellow-400 shrink-0" />
                          <span className="text-neutral-300 truncate">{u.station.name || u.station.id}</span>
                          <span className="text-yellow-400/60 truncate ml-auto">missing: {u.expected}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Submit Button */}
                <button
                  onClick={startMultistationProcessing}
                  disabled={!matchStatus || matchStatus.unmatched.length > 0 || matchStatus.matched.length === 0}
                  className={`w-full py-3 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2 ${
                    matchStatus && matchStatus.unmatched.length === 0 && matchStatus.matched.length > 0
                      ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-900/30 cursor-pointer'
                      : 'bg-neutral-800 text-neutral-500 cursor-not-allowed'
                  }`}
                >
                  <Layers size={16} />
                  {matchStatus && matchStatus.matched.length > 0
                    ? `Process ${matchStatus.matched.length} Stations`
                    : 'Upload JSON and images to begin'}
                </button>
              </div>
            )}
          </div>
        )}

        {/* State: Uploading / Processing */}
        {(appState === 'uploading' || appState === 'processing') && (
          <div className="w-full max-w-xl flex flex-col items-center justify-center gap-8 relative">
            {/* Preview Card */}
            <div className="relative w-64 h-64 rounded-2xl overflow-hidden shadow-2xl border border-white/10">
              {uploadedImage ? (
                <img src={uploadedImage} alt="Input" className="w-full h-full object-cover opacity-80" />
              ) : isMultistation ? (
                <div className="w-full h-full bg-neutral-900 flex flex-col items-center justify-center gap-3">
                  <MapPin size={40} className="text-indigo-400 opacity-60" />
                  <span className="text-sm text-indigo-300/60 font-mono">
                    {matchStatus?.matched?.length || '?'} stations
                  </span>
                </div>
              ) : null}
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />

              {/* Scanning Effect Overlay */}
              <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-500/20 to-transparent h-[50%] animate-scan" />
            </div>

            {/* Mode Badge */}
            {isMultistation && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/20 border border-indigo-500/30 text-indigo-400 text-xs font-medium">
                <MapPin size={14} />
                Multi-Station Capture
              </div>
            )}
            {!isMultistation && is360 && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/20 border border-indigo-500/30 text-indigo-400 text-xs font-medium">
                <Layers size={14} />
                360 Panorama -- Processing 6 cubemap faces
              </div>
            )}

            {/* Progress Area */}
            <div className="w-full space-y-4">
              <div className="flex justify-between text-sm font-medium">
                <span className="text-blue-400 animate-pulse">{appState === 'uploading' ? 'Uploading...' : 'Processing...'}</span>
                <span className="text-neutral-500">{progress}%</span>
              </div>

              <div className="h-1.5 w-full bg-neutral-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>

              <p className="text-center text-neutral-500 text-sm font-mono h-5">
                {statusMessage}
              </p>
            </div>
          </div>
        )}

        {/* State: Complete / Viewer */}
        {appState === 'complete' && (
          <div className="w-full h-full flex flex-col gap-4 animate-fade-in relative">
            {/* Toolbar */}
            <div className="absolute top-4 left-4 right-4 z-50 flex items-center justify-between p-4 bg-neutral-900/60 backdrop-blur-xl border border-white/5 rounded-2xl">
              <div className="flex items-center gap-4">
                <button onClick={resetApp} className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors">
                  <X size={20} />
                </button>
                <div className="h-6 w-px bg-white/10" />
                <h3 className="text-sm font-medium">Result.splat</h3>
                <span className="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/20">Interactive</span>
                {isMultistation && (
                  <span className="text-xs px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/20 flex items-center gap-1">
                    <MapPin size={10} />
                    Multi-Station
                  </span>
                )}
                {!isMultistation && is360 && (
                  <span className="text-xs px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/20 flex items-center gap-1">
                    <Layers size={10} />
                    360
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                <a
                  href={downloadUrl}
                  download="sharp_result.splat"
                  className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors shadow-lg shadow-blue-900/20 text-white"
                >
                  <Download size={14} />
                  <span>Download</span>
                </a>
              </div>
            </div>

            {/* 3D Viewer Container */}
            <div className="w-full h-full relative rounded-3xl overflow-hidden border border-white/5 shadow-2xl bg-black">
              {downloadUrl && (
                <Canvas camera={{ position: [0, 0, 5], fov: 75 }} className={isRecentering ? 'cursor-crosshair' : 'cursor-move'}>
                  <color attach="background" args={['#000000']} />
                  <Splat
                    src={downloadUrl}
                    onClick={(e) => {
                      if (isRecentering) {
                        e.stopPropagation();
                        // Get the intersection point from the event
                        const { x, y, z } = e.point;
                        setCameraTarget({ target: [x, y, z] });
                        setIsRecentering(false);
                      }
                    }}
                  />
                  <OrbitControls
                    makeDefault
                    enabled={!isRecentering}
                    mouseButtons={{
                      LEFT: interactionMode === 'pan' ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE,
                      MIDDLE: THREE.MOUSE.DOLLY,
                      RIGHT: interactionMode === 'pan' ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN
                    }}
                  />
                  <CameraHandler target={cameraTarget} onTargetReached={() => setCameraTarget(null)} />
                </Canvas>
              )}
              <Loader />

              {/* Snap-to-view Bottom Toolbar */}
              <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-2 py-2 bg-neutral-900/80 backdrop-blur-md rounded-full border border-white/10 shadow-xl">
                <ControlButton
                  icon={<Monitor size={18} />}
                  label="Front View"
                  onClick={() => setCameraTarget({ position: [0, 0, 5], target: [0, 0, 0] })}
                />
                <ControlButton
                  icon={<ArrowLeft size={18} />}
                  label="Left View"
                  onClick={() => setCameraTarget({ position: [-5, 0, 0], target: [0, 0, 0] })}
                />
                <ControlButton
                  icon={<ArrowRight size={18} />}
                  label="Right View"
                  onClick={() => setCameraTarget({ position: [5, 0, 0], target: [0, 0, 0] })}
                />
                <ControlButton
                  icon={<ArrowUp size={18} />}
                  label="Top View"
                  onClick={() => setCameraTarget({ position: [0, 5, 0], target: [0, 0, 0] })}
                />
                <div className="w-px h-6 bg-white/10 mx-1" />
                <ControlButton
                  icon={<RotateCcw size={18} />}
                  label="Reset View"
                  onClick={() => {
                    setCameraTarget({ position: [0, 0, 5], target: [0, 0, 0] });
                    setIsRecentering(false);
                  }}
                />
                <ControlButton
                  icon={<Target size={18} />}
                  label={isRecentering ? "Select Point..." : "Recenter"}
                  active={isRecentering}
                  onClick={() => setIsRecentering(!isRecentering)}
                />
                <ControlButton
                  icon={<Hand size={18} />}
                  label={interactionMode === 'pan' ? "Rotate Mode" : "Pan Mode"}
                  active={interactionMode === 'pan'}
                  onClick={() => setInteractionMode(interactionMode === 'pan' ? 'rotate' : 'pan')}
                />
              </div>
            </div>
          </div>
        )}

      </main>

      <style>{`
        @keyframes scan {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(200%); }
        }
        .animate-scan {
          animation: scan 2s linear infinite;
        }
        .animate-fade-in {
          animation: fadeIn 0.5s ease-out forwards;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: scale(0.98); }
          to { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
};

// --- Camera Controller Component ---
const CameraHandler = ({ target, onTargetReached }) => {
  const { controls } = useThree();

  useFrame((state, delta) => {
    if (target) {
      let reached = true;
      const step = 8 * delta; // Faster smoothing

      if (target.position) {
        const destination = new THREE.Vector3(...target.position);
        state.camera.position.lerp(destination, step);
        if (state.camera.position.distanceTo(destination) > 0.01) reached = false;
      }

      if (target.target && controls) {
        const targetVec = new THREE.Vector3(...target.target);
        controls.target.lerp(targetVec, step);
        controls.update();
        if (controls.target.distanceTo(targetVec) > 0.01) reached = false;
      }

      if (reached) {
        onTargetReached?.();
      }
    }
  });

  return null;
};

const ControlButton = ({ icon, label, active, onClick }) => (
  <button
    className={`
      p-3 rounded-full transition-all duration-200 group relative
      ${active ? 'bg-white/20 text-white' : 'hover:bg-white/10 text-neutral-400 hover:text-white'}
    `}
    onClick={onClick}
  >
    {icon}
    <span className="absolute -top-10 left-1/2 -translate-x-1/2 px-2 py-1 bg-black text-xs rounded text-white opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none border border-white/10">
      {label}
    </span>
  </button>
);

export default SharpApp;

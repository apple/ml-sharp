import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Upload, FileImage, Layers, Box, Download, Share2, Maximize2, RotateCw, X, Command, Github } from 'lucide-react';

/**
 * SHARP Web Interface
 * * A modern, fluid frontend for Apple's SHARP (Sharp Monocular View Synthesis) model.
 * * NOTE: This is a frontend client. The SHARP model itself runs in Python/PyTorch.
 * In a real deployment, the `simulateInference` function would be replaced by a 
 * fetch() call to your backend API (e.g., FastAPI/Modal) that returns the .ply file.
 */ 

const SharpApp = () => {
  const [appState, setAppState] = useState('idle'); // idle, uploading, processing, complete
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isHovering, setIsHovering] = useState(false);
  
  // Simulation constants
  const PROCESSING_STEPS = [
    { pct: 10, msg: "Encoding image to latent space..." },
    { pct: 30, msg: "Regressing 3D Gaussian parameters..." },
    { pct: 60, msg: "Refining opacity and spherical harmonics..." },
    { pct: 80, msg: "Assembling scene geometry..." },
    { pct: 100, msg: "Rendering final view..." }
  ];

  const handleFileUpload = (file) => {
    if (!file) return;
    
    // Create preview URL
    const url = URL.createObjectURL(file);
    setUploadedImage(url);
    setAppState('uploading');

    // Simulate upload delay
    setTimeout(() => {
      startProcessing();
    }, 1200);
  };

  const startProcessing = () => {
    setAppState('processing');
    setProgress(0);

    // Simulate the SHARP model inference pipeline
    // In production: await fetch('/api/predict', { method: 'POST', body: formData })
    let currentStep = 0;
    
    const interval = setInterval(() => {
      if (currentStep >= PROCESSING_STEPS.length) {
        clearInterval(interval);
        setTimeout(() => setAppState('complete'), 500);
        return;
      }

      const step = PROCESSING_STEPS[currentStep];
      setProgress(step.pct);
      setStatusMessage(step.msg);
      currentStep++;
    }, 600); // Fast, just like SHARP (<1s inference usually, slowed slightly for UX)
  };

  const resetApp = () => {
    setAppState('idle');
    setUploadedImage(null);
    setProgress(0);
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
              accept="image/*"
            />
            
            <div className="p-4 bg-neutral-800/50 rounded-2xl group-hover:scale-110 transition-transform duration-300 shadow-xl border border-white/5">
              <Upload size={32} className="text-blue-400" />
            </div>
            
            <div className="text-center space-y-2 relative z-10">
              <h2 className="text-2xl font-light text-white">
                Drag and drop your image
              </h2>
              <p className="text-neutral-500 max-w-xs mx-auto">
                Upload a single photo to generate a photorealistic 3D Gaussian Splat scene.
              </p>
            </div>

            {/* Simulated UI Decoration */}
            <div className="absolute bottom-8 flex gap-8 text-neutral-700 font-mono text-xs uppercase tracking-widest opacity-50">
              <span>Single Shot</span>
              <span>•</span>
              <span>1024px Resolution</span>
              <span>•</span>
              <span>.PLY Output</span>
            </div>
          </div>
        )}

        {/* State: Uploading / Processing */}
        {(appState === 'uploading' || appState === 'processing') && (
          <div className="w-full max-w-xl flex flex-col items-center justify-center gap-8 relative">
            {/* Image Preview Card */}
            <div className="relative w-64 h-64 rounded-2xl overflow-hidden shadow-2xl border border-white/10">
              <img src={uploadedImage} alt="Input" className="w-full h-full object-cover opacity-80" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent" />
              
              {/* Scanning Effect Overlay */}
              <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-500/20 to-transparent h-[50%] animate-scan" />
            </div>

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
          <div className="w-full h-full flex flex-col gap-4 animate-fade-in">
            {/* Toolbar */}
            <div className="flex items-center justify-between p-4 bg-neutral-900/60 backdrop-blur-xl border border-white/5 rounded-2xl">
               <div className="flex items-center gap-4">
                 <button onClick={resetApp} className="p-2 hover:bg-white/10 rounded-lg text-neutral-400 hover:text-white transition-colors">
                   <X size={20} />
                 </button>
                 <div className="h-6 w-px bg-white/10" />
                 <h3 className="text-sm font-medium">Result.ply</h3>
                 <span className="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/20">Generated in 0.8s</span>
               </div>

               <div className="flex items-center gap-2">
                 <button className="flex items-center gap-2 px-3 py-1.5 text-sm bg-white/5 hover:bg-white/10 rounded-lg transition-colors border border-white/5">
                   <Share2 size={14} />
                   <span>Share</span>
                 </button>
                 <button className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors shadow-lg shadow-blue-900/20">
                   <Download size={14} />
                   <span>Download .PLY</span>
                 </button>
               </div>
            </div>

            {/* 3D Viewer Container */}
            <div className="flex-1 relative rounded-3xl overflow-hidden border border-white/5 shadow-2xl bg-black">
               {/* Embed 3D Viewer Component */}
               <SimpleSplatViewer />
               
               {/* Viewer Overlay Controls */}
               <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 px-2 py-2 bg-neutral-900/80 backdrop-blur-md rounded-full border border-white/10 shadow-xl">
                  <ControlButton icon={<RotateCw size={18} />} label="Auto-rotate" active />
                  <ControlButton icon={<Layers size={18} />} label="Wireframe" />
                  <ControlButton icon={<Maximize2 size={18} />} label="Fullscreen" />
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

// --- Mock 3D Viewer Component ---
// In a real implementation, you would use a library like 'antimatter15/splat' 
// or '@react-three/drei' with a custom shader for Gaussian Splatting.
// Here we use standard Three.js via CDN in a simple effect for demonstration.

const SimpleSplatViewer = () => {
  const mountRef = useRef(null);

  useEffect(() => {
    // Dynamic import for Three.js to keep file self-contained without npm install steps for the user
    // This is a common pattern for single-file react demos
    const script = document.createElement('script');
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
    script.async = true;
    
    script.onload = () => {
      initThreeJS();
    };
    
    document.body.appendChild(script);

    return () => {
      document.body.removeChild(script);
    };
  }, []);

  const initThreeJS = () => {
    if (!window.THREE || !mountRef.current) return;
    const THREE = window.THREE;

    // Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    scene.fog = new THREE.FogExp2(0x050505, 0.05);

    const camera = new THREE.PerspectiveCamera(75, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 1000);
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.innerHTML = '';
    mountRef.current.appendChild(renderer.domElement);

    // Create "Fake" Gaussian Splat (PointCloud)
    const geometry = new THREE.BufferGeometry();
    const count = 5000;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      // Create a cloudy sphere shape
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      const r = 2 * Math.cbrt(Math.random()); // Cube root for uniform distribution

      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      // Color gradient based on position (Blue/Pink/Purple cyberpunk aesthetic)
      colors[i * 3] = (x + 2) / 4;     // R
      colors[i * 3 + 1] = (y + 2) / 4; // G
      colors[i * 3 + 2] = 1.0;         // B
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Custom material to make points look soft/glowing
    const material = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      depthTest: false,
      transparent: true,
      opacity: 0.8
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // Animation Loop
    let frameId;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      
      particles.rotation.y += 0.002;
      particles.rotation.x += 0.001;

      renderer.render(scene, camera);
    };
    animate();

    // Handle Resize
    const handleResize = () => {
      if (!mountRef.current) return;
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);
  };

  return <div ref={mountRef} className="w-full h-full cursor-move" />;
};

const ControlButton = ({ icon, label, active }) => (
  <button 
    className={`
      p-3 rounded-full transition-all duration-200 group relative
      ${active ? 'bg-white/20 text-white' : 'hover:bg-white/10 text-neutral-400 hover:text-white'}
    `}
  >
    {icon}
    <span className="absolute -top-10 left-1/2 -translate-x-1/2 px-2 py-1 bg-black text-xs rounded text-white opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none border border-white/10">
      {label}
    </span>
  </button>
);

export default SharpApp;

import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import * as THREE from 'three';

// Make THREE available globally for the viewer
window.THREE = THREE;
window.GaussianSplats3D = GaussianSplats3D;

// Viewer state
let viewer = null;
let initialCameraPosition = null;
let initialCameraTarget = null;

// DOM elements
const card = document.querySelector('.card');
const header = document.querySelector('.header');
const viewerContainer = document.getElementById('viewerContainer');
const viewerCanvasContainer = document.getElementById('viewerCanvasContainer');
const viewerFilename = document.getElementById('viewerFilename');
const backBtn = document.getElementById('backBtn');
const resetViewBtn = document.getElementById('resetViewBtn');
const downloadBtn = document.getElementById('downloadBtn');
const controlsHint = document.getElementById('controlsHint');

// Expose showViewer to global scope
window.showViewer = async function (result) {
    // Update filename badge
    viewerFilename.textContent = result.ply_filename;

    // Hide card and show viewer
    card.classList.add('hidden');
    header.classList.add('minimized');
    viewerContainer.classList.add('active');

    // Decode base64 PLY data
    const binaryString = atob(result.ply_data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    const plyBlob = new Blob([bytes], { type: 'application/octet-stream' });
    const plyUrl = URL.createObjectURL(plyBlob);

    // Clean up previous viewer if exists
    if (viewer) {
        viewer.dispose();
        viewer = null;
        // Clear the container
        while (viewerCanvasContainer.firstChild) {
            if (viewerCanvasContainer.firstChild.id !== 'controlsHint') {
                viewerCanvasContainer.removeChild(viewerCanvasContainer.firstChild);
            } else {
                break;
            }
        }
    }

    // Wait for container to be visible and sized
    await new Promise(resolve => setTimeout(resolve, 100));

    try {
        // Create the Gaussian Splat viewer
        viewer = new GaussianSplats3D.Viewer({
            cameraUp: [0, -1, 0],
            initialCameraPosition: [0, 0, -3],
            initialCameraLookAt: [0, 0, 0],
            rootElement: viewerCanvasContainer,
            sharedMemoryForWorkers: false,
            dynamicScene: false,
            sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
            antialiased: true,
        });

        // Load the PLY file - specify format since blob URLs don't have extensions
        await viewer.addSplatScene(plyUrl, {
            splatAlphaRemovalThreshold: 5,
            showLoadingUI: false,
            progressiveLoad: false,
            format: GaussianSplats3D.SceneFormat.Ply,
        });

        viewer.start();

        // Store initial camera state for reset
        if (viewer.camera) {
            initialCameraPosition = viewer.camera.position.clone();
            initialCameraTarget = new THREE.Vector3(0, 0, 0);
        }

        // Hide controls hint after a few seconds
        setTimeout(() => {
            controlsHint.style.opacity = '0';
        }, 5000);

        // Cleanup blob URL after loading
        URL.revokeObjectURL(plyUrl);

    } catch (error) {
        console.error('Error loading Gaussian Splat:', error);
        viewerCanvasContainer.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #ef4444; text-align: center; padding: 2rem;">
                <div>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Failed to load 3D viewer</p>
                    <p style="font-size: 0.875rem; opacity: 0.7;">${error.message}</p>
                </div>
            </div>
        `;
    }
};

// Back button handler
backBtn.addEventListener('click', () => {
    // Clean up viewer
    if (viewer) {
        viewer.dispose();
        viewer = null;
    }

    // Clear the canvas container except for the hint
    const hint = document.getElementById('controlsHint');
    viewerCanvasContainer.innerHTML = '';
    if (hint) {
        hint.style.opacity = '1';
        viewerCanvasContainer.appendChild(hint);
    }

    // Restore upload UI elements
    const dropZone = document.getElementById('dropZone');
    const fileList = document.getElementById('fileList');
    const submitBtn = document.getElementById('submitBtn');
    dropZone.style.display = '';
    fileList.style.display = '';
    submitBtn.style.display = '';

    // Show card and hide viewer
    card.classList.remove('hidden');
    header.classList.remove('minimized');
    viewerContainer.classList.remove('active');
});

// Reset view button handler
resetViewBtn.addEventListener('click', () => {
    if (viewer && viewer.camera && initialCameraPosition) {
        viewer.camera.position.copy(initialCameraPosition);
        viewer.camera.lookAt(initialCameraTarget);
        if (viewer.controls) {
            viewer.controls.target.copy(initialCameraTarget);
            viewer.controls.update();
        }
    }
});

// Download button handler
downloadBtn.addEventListener('click', () => {
    if (window.currentPlyData && window.currentPlyFilename) {
        const binaryString = atob(window.currentPlyData);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = window.currentPlyFilename;
        a.click();
        URL.revokeObjectURL(url);
    }
});

// Handle window resize for the viewer
window.addEventListener('resize', () => {
    if (viewer && viewerContainer.classList.contains('active')) {
        // The viewer should handle resize automatically
    }
});

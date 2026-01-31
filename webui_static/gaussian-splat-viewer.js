/**
 * Gaussian Splat Viewer using Spark.js (THREE.js-based renderer)
 * https://github.com/sparkjsdev/spark
 *
 * Controls:
 * - WASD: Move forward/back/left/right
 * - Q/E: Move up/down
 * - Mouse drag: Look around
 * - Scroll: Adjust move speed
 * - SBS/VR: Side-by-side stereoscopic rendering
 */

class GaussianSplatViewer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.onProgress = options.onProgress || (() => {});
        this.onLoad = options.onLoad || (() => {});
        this.onError = options.onError || ((e) => console.error(e));

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        
        // Single mode
        this.splat = null;
        
        // Video mode
        this.frameMeshes = []; 
        this.currentFrameIndex = -1;

        this.animationId = null;

        // First-person controls state
        this.keys = {};
        this.mouseDown = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.yaw = Math.PI;    // Start facing toward -Z (toward the splat)
        this.pitch = 0;
        this.moveSpeed = 2.0;  // Units per second
        this.lookSpeed = 0.003;
        this.lastTime = performance.now();

        // VR / Stereo State
        this.stereoMode = false;
        this.stereoCamera = null;

        // Create a promise that resolves when init is complete
        this.ready = this.init();
    }

    async init() {
        try {
            console.log('GaussianSplatViewer: Starting initialization...');

            // Import THREE.js and Spark dynamically
            const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js');
            const { SplatMesh } = await import('https://sparkjs.dev/releases/spark/0.1.10/spark.module.js');

            this.THREE = THREE;
            this.SplatMesh = SplatMesh;

            // Setup renderer
            this.renderer = new THREE.WebGLRenderer({
                canvas: this.canvas,
                antialias: true,
                alpha: false
            });
            this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.setClearColor(0x000000, 1);
            
            // Required for partial viewport rendering in VR mode
            this.renderer.setScissorTest(false);

            // Setup scene
            this.scene = new THREE.Scene();

            // Setup camera
            this.camera = new THREE.PerspectiveCamera(
                60,
                this.canvas.clientWidth / this.canvas.clientHeight,
                0.01,
                1000
            );
            this.camera.position.set(0, 0, 3);

            // Setup Stereo Camera for VR
            this.stereoCamera = new THREE.StereoCamera();
            this.stereoCamera.aspect = 0.5;

            // Setup first-person controls
            this.setupControls();

            // Handle resize
            this.resizeObserver = new ResizeObserver(() => this.handleResize());
            this.resizeObserver.observe(this.canvas);

            // Start render loop
            this.animate();

            console.log('GaussianSplatViewer: Initialization complete!');
        } catch (error) {
            console.error('GaussianSplatViewer: Failed to initialize:', error);
            this.onError(error);
            throw error;
        }
    }

    setupControls() {
        this.keyDownHandler = (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            this.keys[e.key.toLowerCase()] = true;
        };
        this.keyUpHandler = (e) => {
            this.keys[e.key.toLowerCase()] = false;
        };

        this.mouseDownHandler = (e) => {
            if (e.target !== this.canvas) return;
            this.mouseDown = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.canvas.style.cursor = 'grabbing';
        };
        this.mouseUpHandler = () => {
            this.mouseDown = false;
            this.canvas.style.cursor = 'grab';
        };
        this.mouseMoveHandler = (e) => {
            if (!this.mouseDown) return;
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            this.yaw -= deltaX * this.lookSpeed;
            this.pitch -= deltaY * this.lookSpeed;
            this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        };

        this.wheelHandler = (e) => {
            if (e.target !== this.canvas) return;
            e.preventDefault();
            if (e.deltaY < 0) this.moveSpeed *= 1.2;
            else this.moveSpeed /= 1.2;
            this.moveSpeed = Math.max(0.1, Math.min(20, this.moveSpeed));
        };

        document.addEventListener('keydown', this.keyDownHandler);
        document.addEventListener('keyup', this.keyUpHandler);
        this.canvas.addEventListener('mousedown', this.mouseDownHandler);
        document.addEventListener('mouseup', this.mouseUpHandler);
        document.addEventListener('mousemove', this.mouseMoveHandler);
        this.canvas.addEventListener('wheel', this.wheelHandler, { passive: false });
        this.canvas.style.cursor = 'grab';
    }

    handleResize() {
        if (!this.renderer || !this.camera) return;
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);

        // Ensure we reset viewport if we exit stereo mode during resize
        if (!this.stereoMode) {
            this.renderer.setScissor(0, 0, width, height);
            this.renderer.setViewport(0, 0, width, height);
        }
    }

    updateControls(deltaTime) {
        if (!this.camera) return;
        const forward = new this.THREE.Vector3(-Math.sin(this.yaw), 0, Math.cos(this.yaw));
        const right = new this.THREE.Vector3(Math.cos(this.yaw), 0, Math.sin(this.yaw));
        const up = new this.THREE.Vector3(0, -1, 0);
        const velocity = new this.THREE.Vector3();
        const speed = this.moveSpeed * deltaTime;

        if (this.keys['w']) velocity.add(forward.clone().multiplyScalar(-speed));
        if (this.keys['s']) velocity.add(forward.clone().multiplyScalar(speed));
        if (this.keys['a']) velocity.add(right.clone().multiplyScalar(speed));
        if (this.keys['d']) velocity.add(right.clone().multiplyScalar(-speed));
        if (this.keys['q'] || this.keys[' ']) velocity.add(up.clone().multiplyScalar(-speed));
        if (this.keys['e'] || this.keys['shift']) velocity.add(up.clone().multiplyScalar(speed));

        this.camera.position.add(velocity);
        const quaternion = new this.THREE.Quaternion();
        const euler = new this.THREE.Euler(-this.pitch, this.yaw, Math.PI, 'YXZ');
        quaternion.setFromEuler(euler);
        this.camera.quaternion.copy(quaternion);
    }

    toggleStereo() {
        if (!this.renderer) return;
        this.stereoMode = !this.stereoMode;
        
        if (this.stereoMode) {
            this.renderer.setScissorTest(true);
        } else {
            this.renderer.setScissorTest(false);
            this.handleResize(); // Reset full viewport
        }
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        const now = performance.now();
        const deltaTime = (now - this.lastTime) / 1000;
        this.lastTime = now;
        
        this.updateControls(deltaTime);
        
        if (this.renderer && this.scene && this.camera) {
            if (this.stereoMode && this.stereoCamera) {
                // VR / SBS Mode
                const width = this.canvas.width;
                const height = this.canvas.height;

                // CRITICAL FIX: Manually update the main camera matrix
                // because we aren't passing it to renderer.render(), so Three.js
                // doesn't automatically update it. This fixes WASD movement in SBS mode.
                this.camera.updateMatrixWorld();

                // Sync the stereo camera rig with the main camera position/rotation
                this.stereoCamera.update(this.camera);

                // Render Left Eye
                this.renderer.setScissor(0, 0, width / 2, height);
                this.renderer.setViewport(0, 0, width / 2, height);
                this.renderer.render(this.scene, this.stereoCamera.cameraL);

                // Render Right Eye
                this.renderer.setScissor(width / 2, 0, width / 2, height);
                this.renderer.setViewport(width / 2, 0, width / 2, height);
                this.renderer.render(this.scene, this.stereoCamera.cameraR);
            } else {
                // Standard Single Mode
                this.renderer.render(this.scene, this.camera);
            }
        }
    }

    /**
     * Preloads ALL frames into memory.
     * VISUAL UPDATE: Shows frames as they load.
     */
    async preloadFrames(urls, progressCallback) {
        await this.ready;
        this.clearScene(); 

        // Cleanup existing frames
        if (this.frameMeshes.length > 0) {
            this.frameMeshes.forEach(mesh => {
                this.scene.remove(mesh);
                if (mesh.dispose) mesh.dispose();
            });
            this.frameMeshes = [];
        }

        console.log(`GaussianSplatViewer: Preloading ${urls.length} frames...`);

        try {
            for (let i = 0; i < urls.length; i++) {
                const url = urls[i];
                
                const mesh = new this.SplatMesh({
                    url: url,
                    visible: false
                });

                // Wait for parsing
                await mesh.loadPromise;
                
                // VISUAL FEEDBACK: 
                // 1. Hide previous frame if it exists
                if (this.currentFrameIndex >= 0 && this.frameMeshes[this.currentFrameIndex]) {
                    this.frameMeshes[this.currentFrameIndex].visible = false;
                }

                // 2. Add new frame and make visible immediately
                mesh.visible = true;
                this.scene.add(mesh);
                this.frameMeshes.push(mesh);
                this.currentFrameIndex = i;

                // Center camera on first frame only
                if (i === 0) {
                    this.centerCameraOnSplat();
                }

                if (progressCallback) {
                    progressCallback(i + 1, urls.length);
                }
            }

            this.onLoad();

        } catch (error) {
            console.error("Error preloading frames:", error);
            this.onError(error);
        }
    }

    /**
     * Instant frame switching (since they are all in VRAM).
     */
    showFrame(index) {
        if (!this.frameMeshes || this.frameMeshes.length === 0) return;
        if (index < 0 || index >= this.frameMeshes.length) return;

        // Hide current
        if (this.currentFrameIndex !== -1 && this.frameMeshes[this.currentFrameIndex]) {
            this.frameMeshes[this.currentFrameIndex].visible = false;
        }

        // Show next
        this.frameMeshes[index].visible = true;
        this.currentFrameIndex = index;
    }

    /**
     * Legacy single file loader
     */
    async loadPly(url, centerCamera = true) {
        await this.ready;
        
        // Clear video data if any
        if (this.frameMeshes.length > 0) {
            this.frameMeshes.forEach(m => {
                this.scene.remove(m);
                if(m.dispose) m.dispose();
            });
            this.frameMeshes = [];
            this.currentFrameIndex = -1;
        }

        if (centerCamera) this.onProgress(10);

        try {
            const newSplat = new this.SplatMesh({
                url: url,
                onProgress: (progress) => {
                    if (centerCamera) this.onProgress(20 + progress * 70);
                }
            });

            await newSplat.loadPromise;

            if (centerCamera) this.onProgress(95);

            this.scene.add(newSplat);

            if (this.splat) {
                const old = this.splat;
                setTimeout(() => {
                    this.scene.remove(old);
                    if(old.dispose) old.dispose();
                }, 100);
            }
            this.splat = newSplat;

            if (centerCamera) {
                this.centerCameraOnSplat();
            }

            if (centerCamera) this.onProgress(100);
            this.onLoad();

        } catch (error) {
            console.error('GaussianSplatViewer: Failed to load PLY:', error);
            this.onError(error);
        }
    }

    clearScene() {
        if (this.splat) {
            this.scene.remove(this.splat);
            if (this.splat.dispose) this.splat.dispose();
            this.splat = null;
        }
    }

    centerCameraOnSplat() {
        if (!this.camera) return;
        const distance = 3.0;
        this.camera.position.set(0, 0, distance);
        this.yaw = Math.PI;
        this.pitch = 0;
    }

    resetCamera() {
        this.centerCameraOnSplat();
    }

    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        document.removeEventListener('keydown', this.keyDownHandler);
        document.removeEventListener('keyup', this.keyUpHandler);
        this.canvas.removeEventListener('mousedown', this.mouseDownHandler);
        document.removeEventListener('mouseup', this.mouseUpHandler);
        document.removeEventListener('mousemove', this.mouseMoveHandler);
        this.canvas.removeEventListener('wheel', this.wheelHandler);

        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        
        this.clearScene();
        if (this.frameMeshes.length > 0) {
            this.frameMeshes.forEach(m => {
                this.scene.remove(m);
                if(m.dispose) m.dispose();
            });
            this.frameMeshes = [];
        }

        if (this.renderer) {
            this.renderer.dispose();
            this.renderer = null;
        }
        this.scene = null;
        this.camera = null;
        this.stereoCamera = null;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GaussianSplatViewer;
}
// Global state for PLY data (on window so module script can access)
window.currentPlyData = null;
window.currentPlyFilename = null;

// Particle system for background
const canvas = document.getElementById('particleCanvas');
const ctx = canvas.getContext('2d');
let particles = [];
let animationId;
let mouseX = 0, mouseY = 0;

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

class Particle {
    constructor() {
        this.reset();
    }

    reset() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.z = Math.random() * 1000;
        this.baseSize = Math.random() * 2 + 0.5;
        this.color = this.getColor();
        this.vx = (Math.random() - 0.5) * 0.3;
        this.vy = (Math.random() - 0.5) * 0.3;
        this.vz = (Math.random() - 0.5) * 2;
    }

    getColor() {
        const colors = [
            { r: 99, g: 102, b: 241 },   // indigo
            { r: 168, g: 85, b: 247 },   // purple
            { r: 236, g: 72, b: 153 },   // pink
            { r: 59, g: 130, b: 246 },   // blue
            { r: 139, g: 92, b: 246 },   // violet
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    update() {
        // Mouse interaction
        const dx = mouseX - this.x;
        const dy = mouseY - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 150) {
            const force = (150 - dist) / 150;
            this.vx -= (dx / dist) * force * 0.5;
            this.vy -= (dy / dist) * force * 0.5;
        }

        this.x += this.vx;
        this.y += this.vy;
        this.z += this.vz;

        // Damping
        this.vx *= 0.99;
        this.vy *= 0.99;

        // Wrap around
        if (this.x < 0) this.x = canvas.width;
        if (this.x > canvas.width) this.x = 0;
        if (this.y < 0) this.y = canvas.height;
        if (this.y > canvas.height) this.y = 0;
        if (this.z < 0 || this.z > 1000) this.vz *= -1;
    }

    draw() {
        const perspective = 1000 / (1000 + this.z);
        const size = this.baseSize * perspective * 3;
        const isLightMode = document.documentElement.getAttribute('data-theme') === 'light';
        const alpha = perspective * (isLightMode ? 0.4 : 0.6);

        ctx.beginPath();
        const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, size * 2);
        gradient.addColorStop(0, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${alpha})`);
        gradient.addColorStop(1, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, 0)`);
        ctx.fillStyle = gradient;
        ctx.arc(this.x, this.y, size * 2, 0, Math.PI * 2);
        ctx.fill();
    }
}

function initParticles() {
    particles = [];
    const count = Math.min(200, Math.floor((canvas.width * canvas.height) / 8000));
    for (let i = 0; i < count; i++) {
        particles.push(new Particle());
    }
}

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    particles.forEach(p => {
        p.update();
        p.draw();
    });

    // Draw connections
    const isLightMode = document.documentElement.getAttribute('data-theme') === 'light';
    ctx.strokeStyle = isLightMode ? 'rgba(99, 102, 241, 0.08)' : 'rgba(255, 255, 255, 0.02)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 100) {
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.stroke();
            }
        }
    }

    animationId = requestAnimationFrame(animate);
}

window.addEventListener('resize', () => {
    resizeCanvas();
    initParticles();
});

document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
});

resizeCanvas();
initParticles();
animate();

// Theme toggle functionality
const themeToggle = document.getElementById('themeToggle');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
}

function getPreferredTheme() {
    const stored = localStorage.getItem('theme');
    if (stored) return stored;
    return prefersDark.matches ? 'dark' : 'light';
}

// Initialize theme
setTheme(getPreferredTheme());

themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    setTheme(current === 'light' ? 'dark' : 'light');
});

// Listen for system theme changes
prefersDark.addEventListener('change', (e) => {
    if (!localStorage.getItem('theme')) {
        setTheme(e.matches ? 'dark' : 'light');
    }
});

// Upload functionality
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const form = document.getElementById('uploadForm');
const loaderContainer = document.getElementById('loaderContainer');
const results = document.getElementById('results');
const submitBtn = document.getElementById('submitBtn');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    fileInput.files = e.dataTransfer.files;
    updateFileList();
});

fileInput.addEventListener('change', updateFileList);

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function updateFileList() {
    fileList.innerHTML = '';

    if (fileInput.files.length > 0) {
        dropZone.classList.add('has-files');
    } else {
        dropZone.classList.remove('has-files');
    }

    for (const file of fileInput.files) {
        const div = document.createElement('div');
        div.className = 'file-item';

        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = div.querySelector('.file-preview');
            if (preview) preview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        div.innerHTML = `
            <img class="file-preview" src="" alt="">
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${formatFileSize(file.size)}</div>
            </div>
        `;
        fileList.appendChild(div);
    }
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (fileInput.files.length === 0) return;

    submitBtn.disabled = true;
    dropZone.style.display = 'none';
    fileList.style.display = 'none';
    submitBtn.style.display = 'none';
    loaderContainer.classList.add('active');
    results.innerHTML = '';

    const formData = new FormData();
    for (const file of fileInput.files) {
        formData.append('files', file);
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();

            if (data.results && data.results.length > 0) {
                const result = data.results[0];

                if (result.error) {
                    showError(result.error);
                } else {
                    // Store the PLY data and show the viewer
                    window.currentPlyData = result.ply_data;
                    window.currentPlyFilename = result.ply_filename;

                    // Show viewer (check if module loaded)
                    if (typeof window.showViewer === 'function') {
                        window.showViewer(result);
                    } else {
                        // Fallback: offer download if viewer module failed to load
                        showError('3D viewer failed to load. Click the download button to get your PLY file.');
                        // Trigger download
                        downloadPly(result.ply_data, result.ply_filename);
                    }
                }
            }
        } else {
            const error = await response.text();
            showError(error);
            // Restore upload UI on error
            dropZone.style.display = '';
            fileList.style.display = '';
            submitBtn.style.display = '';
        }
    } catch (err) {
        showError(err.message);
        // Restore upload UI on error
        dropZone.style.display = '';
        fileList.style.display = '';
        submitBtn.style.display = '';
    } finally {
        submitBtn.disabled = false;
        loaderContainer.classList.remove('active');
    }
});

function showError(message) {
    results.innerHTML = `
        <div class="result-item" style="background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.2);">
            <div class="success-icon" style="background: rgba(239, 68, 68, 0.2);">
                <svg viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </div>
            <div class="result-text">
                <div class="result-title" style="color: #ef4444;">Error</div>
                <div class="result-desc">${message}</div>
            </div>
        </div>
    `;
}

function downloadPly(plyData, filename) {
    const binaryString = atob(plyData);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Automatic Dual Authentication Handler
 */

class AutoDualAuth {
    constructor() {
        // Elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.waveformCanvas = document.getElementById('waveform');
        this.waveformCtx = this.waveformCanvas.getContext('2d');
        this.faceBox = document.getElementById('faceBox');
        
        this.authBtn = document.getElementById('authBtn');
        this.stopBtn = document.getElementById('stopBtn');
        
        this.cameraStatus = document.getElementById('cameraStatus');
        this.cameraText = document.getElementById('cameraText');
        this.audioStatus = document.getElementById('audioStatus');
        this.audioText = document.getElementById('audioText');
        
        this.micPlaceholder = document.getElementById('micPlaceholder');
        this.loading = document.getElementById('loading');
        this.result = document.getElementById('result');
        this.instructionDialog = document.getElementById('instructionDialog');
        this.startAuthBtn = document.getElementById('startAuthBtn');
        
        // State
        this.stream = null;
        this.audioRecorder = null;
        this.recordedBlob = null;
        this.visualizerAnalyser = null;
        this.animationId = null;
        this.faceDetectionInterval = null;
        this.modelsLoaded = false;
        this.isListeningForVoice = false;
        this.voiceDetectionInterval = null;
        
        this.init();
    }
    
    init() {
        this.authBtn.addEventListener('click', () => this.showInstructions());
        this.startAuthBtn.addEventListener('click', () => this.startVoiceActivatedAuth());
        this.stopBtn.addEventListener('click', () => this.stop());
        
        // Load face detection models
        this.loadFaceDetectionModels();
        
        // Auto-start camera
        this.start();
    }
    
    async loadFaceDetectionModels() {
        try {
            const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
            await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
            this.modelsLoaded = true;
            console.log('Face detection models loaded');
        } catch (error) {
            console.error('Error loading face detection models:', error);
        }
    }
    
    async start() {
        try {
            // Start camera
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = this.stream;
            
            // Wait for video to be ready
            await new Promise(resolve => {
                this.video.onloadedmetadata = () => {
                    resolve();
                };
            });
            
            // Start face detection
            this.startFaceDetection();
            
            // Update UI
            this.cameraStatus.classList.add('active');
            this.cameraText.textContent = 'Buscando rostro...';
            this.authBtn.disabled = true; // Disabled until face is detected
            
        } catch (error) {
            console.error('Error starting camera:', error);
            this.showResult(false, 'Error al acceder a la cámara', error.message);
        }
    }
    
    startFaceDetection() {
        if (!this.modelsLoaded) {
            console.log('Face detection models not loaded yet');
            return;
        }
        
        this.faceDetectionInterval = setInterval(async () => {
            await this.detectFace();
        }, 100); // Detect every 100ms
    }
    
    async detectFace() {
        if (!this.video || this.video.paused) return;
        
        try {
            const detections = await faceapi.detectSingleFace(
                this.video,
                new faceapi.TinyFaceDetectorOptions()
            );
            
            if (detections) {
                const box = detections.box;
                const videoRect = this.video.getBoundingClientRect();
                const containerRect = this.video.parentElement.getBoundingClientRect();
                
                // Calculate scale
                const scaleX = videoRect.width / this.video.videoWidth;
                const scaleY = videoRect.height / this.video.videoHeight;
                
                // Position and size the face box
                this.faceBox.style.left = `${box.x * scaleX}px`;
                this.faceBox.style.top = `${box.y * scaleY}px`;
                this.faceBox.style.width = `${box.width * scaleX}px`;
                this.faceBox.style.height = `${box.height * scaleY}px`;
                this.faceBox.style.display = 'block';
                
                // Enable auth button when face is detected
                this.authBtn.disabled = false;
                this.cameraText.textContent = 'Rostro detectado - Listo para autenticar';
            } else {
                this.faceBox.style.display = 'none';
                
                // Disable auth button when no face detected
                this.authBtn.disabled = true;
                this.cameraText.textContent = 'Buscando rostro...';
            }
        } catch (error) {
            // Silently fail
            this.authBtn.disabled = true;
        }
    }
    
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
            this.stream = null;
        }
        
        if (this.faceDetectionInterval) {
            clearInterval(this.faceDetectionInterval);
            this.faceDetectionInterval = null;
        }
        
        if (this.audioRecorder && this.audioRecorder.isRecording) {
            this.audioRecorder.stop();
        }
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        this.faceBox.style.display = 'none';
        this.cameraStatus.classList.remove('active');
        this.cameraText.textContent = 'Cámara inactiva';
        this.audioStatus.classList.remove('recording');
        this.audioText.textContent = 'Listo para grabar';
        this.micPlaceholder.style.display = 'block';
        
        this.result.classList.remove('show');
        
        // Restart camera
        this.start();
    }
    
    showInstructions() {
        this.instructionDialog.classList.add('show');
    }
    
    async startVoiceActivatedAuth() {
        // Hide instruction dialog
        this.instructionDialog.classList.remove('show');
        
        if (!this.stream) return;
        
        try {
            this.authBtn.disabled = true;
            this.result.classList.remove('show');
            
            // Start audio recorder to detect voice
            this.audioRecorder = new AudioRecorder();
            await this.audioRecorder.start();
            
            this.audioStatus.classList.add('recording');
            this.audioText.textContent = 'Esperando que hables...';
            this.micPlaceholder.style.display = 'none';
            
            this.setupVisualizer();
            
            // Start listening for voice
            this.isListeningForVoice = true;
            await this.waitForVoiceAndRecord();
            
        } catch (error) {
            console.error('Error during authentication:', error);
            this.showResult(false, 'Error de Autenticación', error.message);
            this.authBtn.disabled = false;
        }
    }
    
    async waitForVoiceAndRecord() {
        return new Promise(async (resolve) => {
            const context = this.audioRecorder.getAudioContext();
            const source = context.createMediaStreamSource(this.audioRecorder.mediaStream);
            const analyser = context.createAnalyser();
            analyser.fftSize = 2048;
            source.connect(analyser);
            
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            const VOICE_THRESHOLD = 30; // Adjust sensitivity
            const CHECK_INTERVAL = 100; // Check every 100ms
            
            const checkVoice = () => {
                if (!this.isListeningForVoice) {
                    resolve();
                    return;
                }
                
                analyser.getByteFrequencyData(dataArray);
                
                // Calculate average volume
                let sum = 0;
                for (let i = 0; i < bufferLength; i++) {
                    sum += dataArray[i];
                }
                const average = sum / bufferLength;
                
                // If voice detected, start recording
                if (average > VOICE_THRESHOLD) {
                    this.isListeningForVoice = false;
                    this.audioText.textContent = 'Grabando (3s)... Habla ahora';
                    
                    // Record for 3 seconds
                    setTimeout(async () => {
                        await this.finishRecordingAndAuthenticate();
                        resolve();
                    }, 3000);
                } else {
                    // Keep checking
                    setTimeout(checkVoice, CHECK_INTERVAL);
                }
            };
            
            checkVoice();
        });
    }
    
    async finishRecordingAndAuthenticate() {
        // Stop recording
        this.audioRecorder.stop();
        this.recordedBlob = this.audioRecorder.getWAVBlob();
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        this.audioStatus.classList.remove('recording');
        this.audioText.textContent = 'Procesando...';
        
        // Capture image
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.ctx.drawImage(this.video, 0, 0);
        
        const imageBlob = await new Promise(resolve => {
            this.canvas.toBlob(resolve, 'image/jpeg', 0.95);
        });
        
        // Show loading
        this.loading.classList.add('show');
        
        try {
            // Send to backend
            await this.sendToBackend(imageBlob, this.recordedBlob);
        } catch (error) {
            console.error('Error during authentication:', error);
            this.showResult(false, 'Error de Autenticación', error.message);
        } finally {
            this.loading.classList.remove('show');
            this.authBtn.disabled = false;
            this.audioText.textContent = 'Listo para grabar';
            this.micPlaceholder.style.display = 'block';
        }
    }
    
    async authenticate() {
        if (!this.stream) return;
        
        try {
            this.authBtn.disabled = true;
            this.result.classList.remove('show');
            
            // Start recording audio
            this.audioRecorder = new AudioRecorder();
            await this.audioRecorder.start();
            
            this.audioStatus.classList.add('recording');
            this.audioText.textContent = 'Grabando (3s)... Habla ahora';
            this.micPlaceholder.style.display = 'none';
            
            this.setupVisualizer();
            
            // Wait 3 seconds
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            // Stop recording
            this.audioRecorder.stop();
            this.recordedBlob = this.audioRecorder.getWAVBlob();
            
            if (this.animationId) {
                cancelAnimationFrame(this.animationId);
            }
            
            this.audioStatus.classList.remove('recording');
            this.audioText.textContent = 'Procesando...';
            
            // Capture image
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx.drawImage(this.video, 0, 0);
            
            const imageBlob = await new Promise(resolve => {
                this.canvas.toBlob(resolve, 'image/jpeg', 0.95);
            });
            
            // Show loading
            this.loading.classList.add('show');
            
            // Send to backend
            await this.sendToBackend(imageBlob, this.recordedBlob);
            
        } catch (error) {
            console.error('Error during authentication:', error);
            this.showResult(false, 'Error de Autenticación', error.message);
        } finally {
            this.loading.classList.remove('show');
            this.authBtn.disabled = false;
            this.audioText.textContent = 'Listo para grabar';
            this.micPlaceholder.style.display = 'block';
        }
    }
    
    async sendToBackend(imageBlob, audioBlob) {
        const formData = new FormData();
        formData.append('image', imageBlob, 'capture.jpg');
        formData.append('audio', audioBlob, 'capture.wav');
        
        try {
            const response = await fetch('/api/authenticate_dual/', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            console.log('Result:', result);
            
            this.displayResult(result);
            
        } catch (error) {
            this.showResult(false, 'Error de Conexión', error.message);
        }
    }
    
    displayResult(result) {
        const success = result.success;
        const message = result.message;
        
        // Capitalize identity names properly and translate
        const capitalize = (str) => {
            if (!str || str === 'No detectado') return str;
            // Translate unknown to Desconocido
            if (str.toLowerCase() === 'unknown') return 'Desconocido';
            // Capitalize first letter
            return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
        };
        
        const faceId = capitalize(result.face_result?.identity) || 'No detectado';
        const faceConf = result.face_result?.confidence || 0;
        const voiceId = capitalize(result.voice_result?.identity) || 'No detectado';
        const voiceConf = (result.voice_result?.confidence || 0) * 100;
        
        const iconClass = success ? 'success' : 'error';
        const icon = success ? '✓' : '✕';
        const title = success ? 'Autenticación Exitosa' : 'Acceso Denegado';
        
        // Hide camera and audio sections
        const sections = document.querySelectorAll('.section');
        sections.forEach(section => section.style.display = 'none');
        
        // Hide controls
        document.querySelector('.controls').style.display = 'none';
        
        let html = `
            <div class="result-content">
                <div class="result-icon ${iconClass}">${icon}</div>
                <h2 class="result-title">${title}</h2>
                <p class="result-message">${message}</p>
        `;
        
        if (result.face_result || result.voice_result) {
            html += `
                <button class="details-toggle" onclick="this.nextElementSibling.classList.toggle('show'); this.textContent = this.nextElementSibling.classList.contains('show') ? '▼ Ocultar detalles' : '▶ Ver detalles'">▶ Ver detalles</button>
                <div class="result-details">
                    <div class="detail-card">
                        <div class="detail-label">Reconocimiento Facial</div>
                        <div class="detail-value">${faceId}</div>
                        <div class="detail-confidence">Confianza: ${faceConf.toFixed(1)}%</div>
                    </div>
                    <div class="detail-card">
                        <div class="detail-label">Análisis de Voz</div>
                        <div class="detail-value">${voiceId}</div>
                        <div class="detail-confidence">Confianza: ${voiceConf.toFixed(1)}%</div>
                    </div>
                </div>
            `;
        }
        
        html += `
                <button class="btn btn-primary" onclick="location.reload()" style="margin-top: 20px; width: 100%;">
                    🔄 Reintentar
                </button>
            </div>
        `;
        
        this.result.innerHTML = html;
        this.result.classList.add('show');
    }
    
    showResult(success, title, message) {
        const iconClass = success ? 'success' : 'error';
        const icon = success ? '✓' : '✕';
        
        const html = `
            <div class="result-content">
                <div class="result-icon ${iconClass}">${icon}</div>
                <h2 class="result-title">${title}</h2>
                <p class="result-message">${message}</p>
            </div>
        `;
        
        this.result.innerHTML = html;
        this.result.classList.add('show');
    }
    
    setupVisualizer() {
        const context = this.audioRecorder.getAudioContext();
        const source = context.createMediaStreamSource(this.audioRecorder.mediaStream);
        this.visualizerAnalyser = context.createAnalyser();
        this.visualizerAnalyser.fftSize = 2048;
        source.connect(this.visualizerAnalyser);
        
        const bufferLength = this.visualizerAnalyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            this.animationId = requestAnimationFrame(draw);
            this.visualizerAnalyser.getByteTimeDomainData(dataArray);
            
            this.waveformCtx.fillStyle = '#2d3748';
            this.waveformCtx.fillRect(0, 0, this.waveformCanvas.width, this.waveformCanvas.height);
            
            this.waveformCtx.lineWidth = 2;
            this.waveformCtx.strokeStyle = '#667eea';
            this.waveformCtx.beginPath();
            
            const sliceWidth = this.waveformCanvas.width / bufferLength;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * this.waveformCanvas.height / 2;
                
                if (i === 0) {
                    this.waveformCtx.moveTo(x, y);
                } else {
                    this.waveformCtx.lineTo(x, y);
                }
                
                x += sliceWidth;
            }
            
            this.waveformCtx.lineTo(this.waveformCanvas.width, this.waveformCanvas.height / 2);
            this.waveformCtx.stroke();
        };
        
        draw();
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    new AutoDualAuth();
});

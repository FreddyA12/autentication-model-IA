/**
 * Dual Authentication Handler (Face + Voice)
 */

class DualAuthApp {
    constructor() {
        // UI Elements
        this.video = document.getElementById('dualVideo');
        this.canvas = document.getElementById('dualCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.startBtn = document.getElementById('startDualAuth');
        this.stopBtn = document.getElementById('stopDualAuth');
        this.captureBtn = document.getElementById('captureDualBtn');
        
        this.statusElement = document.getElementById('dualStatus');
        
        this.waveformCanvas = document.getElementById('dualWaveform');
        this.waveformCtx = this.waveformCanvas ? this.waveformCanvas.getContext('2d') : null;

        // State
        this.stream = null;
        this.audioRecorder = null;
        this.isRecording = false;
        this.visualizerAnalyser = null;
        this.animationId = null;
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        if(this.startBtn) this.startBtn.addEventListener('click', () => this.start());
        if(this.stopBtn) this.stopBtn.addEventListener('click', () => this.stop());
        if(this.captureBtn) this.captureBtn.addEventListener('click', () => this.captureAndAuthenticate());
    }
    
    async start() {
        try {
            // 1. Start Camera
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                audio: false // We handle audio separately with AudioRecorder
            });
            
            this.video.srcObject = this.stream;
            
            // 2. Prepare Audio Recorder
            this.audioRecorder = new AudioRecorder();
            
            // Update UI
            this.startBtn.style.display = 'none';
            this.stopBtn.style.display = 'inline-block';
            this.captureBtn.disabled = false;
            this.updateStatus('Listo. Presiona "Capturar y Autenticar" para comenzar.', 'info');
            
        } catch (error) {
            console.error('Error starting dual auth:', error);
            this.updateStatus('Error al acceder a dispositivos: ' + error.message, 'error');
        }
    }
    
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
            this.stream = null;
        }
        
        if (this.audioRecorder && this.audioRecorder.isRecording) {
            this.audioRecorder.stop();
        }
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        this.startBtn.style.display = 'inline-block';
        this.stopBtn.style.display = 'none';
        this.captureBtn.disabled = true;
        this.updateStatus('Sistema detenido', 'info');
    }
    
    async captureAndAuthenticate() {
        if (!this.stream) return;
        
        try {
            this.captureBtn.disabled = true;
            this.updateStatus('Grabando audio (3s)... Por favor habla.', 'warning');
            
            // 1. Start Audio Recording
            await this.audioRecorder.start();
            this.setupVisualizer();
            
            // 2. Wait 3 seconds
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            // 3. Stop Audio Recording
            this.audioRecorder.stop();
            const audioBlob = this.audioRecorder.getWAVBlob();
            
            // 4. Capture Image
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx.drawImage(this.video, 0, 0);
            
            const imageBlob = await new Promise(resolve => {
                this.canvas.toBlob(resolve, 'image/jpeg', 0.95);
            });
            
            this.updateStatus('Analizando...', 'info');
            
            // 5. Send to Backend
            await this.sendToBackend(imageBlob, audioBlob);
            
        } catch (error) {
            console.error('Error in capture sequence:', error);
            this.updateStatus('Error: ' + error.message, 'error');
        } finally {
            this.captureBtn.disabled = false;
            if (this.animationId) cancelAnimationFrame(this.animationId);
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
            console.log('Dual Auth Result:', result);
            
            this.displayResult(result);
            
        } catch (error) {
            this.updateStatus('Error de red: ' + error.message, 'error');
        }
    }
    
    displayResult(result) {
        const resultDiv = document.getElementById('dualResultContent');
        const resultCard = document.getElementById('dualResultCard');
        
        resultCard.style.display = 'block';
        
        let html = '';
        if (result.success) {
            html = `<div class="success-message" style="color: #10b981; text-align: center;">
                        <h3>✅ ¡Autenticado!</h3>
                        <p>${result.message}</p>
                    </div>`;
        } else {
            html = `<div class="error-message" style="color: #ef4444; text-align: center;">
                        <h3>❌ Acceso Denegado</h3>
                        <p>${result.message}</p>
                    </div>`;
        }
        
        // Details
        const faceConf = result.face_result.confidence ? result.face_result.confidence.toFixed(1) : 0;
        const voiceConf = result.voice_result.confidence ? (result.voice_result.confidence * 100).toFixed(1) : 0;
        
        html += `<div class="details" style="margin-top: 15px; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
                    <p><strong>👤 Cara:</strong> ${result.face_result.identity || 'No detectada'} (${faceConf}%)</p>
                    <p><strong>🎤 Voz:</strong> ${result.voice_result.identity || 'No detectada'} (${voiceConf}%)</p>
                 </div>`;
                 
        resultDiv.innerHTML = html;
        this.updateStatus(result.success ? 'Autenticación Exitosa' : 'Falló la autenticación', result.success ? 'success' : 'error');
    }
    
    updateStatus(msg, type) {
        if(this.statusElement) {
            this.statusElement.textContent = msg;
            // Reset classes
            this.statusElement.className = 'status-text';
            if (type) this.statusElement.classList.add(type);
        }
    }

    setupVisualizer() {
        if (!this.waveformCanvas) return;
        
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
            
            this.waveformCtx.fillStyle = 'rgba(255, 255, 255, 0.1)';
            this.waveformCtx.fillRect(0, 0, this.waveformCanvas.width, this.waveformCanvas.height);
            this.waveformCtx.lineWidth = 2;
            this.waveformCtx.strokeStyle = '#22d3ee';
            this.waveformCtx.beginPath();
            
            const sliceWidth = this.waveformCanvas.width / bufferLength;
            let x = 0;
            
            for(let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * this.waveformCanvas.height / 2;
                
                if(i === 0) this.waveformCtx.moveTo(x, y);
                else this.waveformCtx.lineTo(x, y);
                
                x += sliceWidth;
            }
            
            this.waveformCtx.lineTo(this.waveformCanvas.width, this.waveformCanvas.height/2);
            this.waveformCtx.stroke();
        };
        
        draw();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Check if elements exist before initializing (since they are in a tab)
    if(document.getElementById('dualVideo')) {
        window.dualAuthApp = new DualAuthApp();
    }
});

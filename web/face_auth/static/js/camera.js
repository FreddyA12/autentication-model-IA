/**
 * Camera and Face Recognition Handler
 */

class FaceAuthApp {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.startCameraBtn = document.getElementById('startCamera');
        this.stopCameraBtn = document.getElementById('stopCamera');
        this.captureBtn = document.getElementById('captureBtn');
        this.cameraStatus = document.getElementById('cameraStatus');
        
        this.stream = null;
        this.isCapturing = false;
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
        this.captureBtn.addEventListener('click', () => this.captureAndPredict());
    }
    
    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = this.stream;
            
            // Update UI
            this.updateCameraStatus(true);
            this.startCameraBtn.disabled = true;
            this.stopCameraBtn.disabled = false;
            this.captureBtn.disabled = false;
            
            this.showNotification('Cámara activada correctamente', 'success');
            
        } catch (error) {
            console.error('Error al acceder a la cámara:', error);
            this.showNotification('No se pudo acceder a la cámara. Verifica los permisos.', 'error');
            this.showError('No se pudo acceder a la cámara', error.message);
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
            this.stream = null;
        }
        
        // Update UI
        this.updateCameraStatus(false);
        this.startCameraBtn.disabled = false;
        this.stopCameraBtn.disabled = true;
        this.captureBtn.disabled = true;
        
        this.showNotification('Cámara desactivada', 'info');
    }
    
    updateCameraStatus(isActive) {
        const statusIcon = this.cameraStatus.querySelector('.status-icon');
        const statusText = this.cameraStatus.querySelector('.status-text');
        
        if (isActive) {
            this.cameraStatus.classList.add('active');
            statusIcon.textContent = '🟢';
            statusText.textContent = 'Cámara activa';
        } else {
            this.cameraStatus.classList.remove('active');
            statusIcon.textContent = '📷';
            statusText.textContent = 'Cámara desactivada';
        }
    }
    
    async captureAndPredict() {
        if (this.isCapturing) return;
        
        try {
            this.isCapturing = true;
            this.captureBtn.disabled = true;
            
            // Show loading state
            this.showLoading();
            
            // Capture frame from video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            this.ctx.drawImage(this.video, 0, 0);
            
            // Convert canvas to blob
            const blob = await new Promise(resolve => {
                this.canvas.toBlob(resolve, 'image/jpeg', 0.95);
            });
            
            // Send to API
            const result = await this.sendImageToAPI(blob);
            
            // Display result
            this.displayResult(result);
            
        } catch (error) {
            console.error('Error en captura:', error);
            this.showError('Error al procesar la imagen', error.message);
        } finally {
            this.isCapturing = false;
            this.captureBtn.disabled = false;
        }
    }
    
    async sendImageToAPI(blob) {
        const formData = new FormData();
        formData.append('image', blob, 'capture.jpg');
        
        const response = await fetch('/api/predict/', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    displayResult(result) {
        // Hide all states
        this.hideAllStates();
        
        if (result.success) {
            // Show success state
            this.showSuccess(result);
        } else if (result.identity === null) {
            // No face detected
            this.showError('No se detectó ningún rostro', result.message);
        } else if (result.identity === 'DESCONOCIDO') {
            // Unknown person
            this.showUnknown(result);
        } else {
            // Other errors
            this.showError('Error en el reconocimiento', result.message);
        }
    }
    
    showLoading() {
        this.hideAllStates();
        document.getElementById('loadingState').style.display = 'flex';
    }
    
    showSuccess(result) {
        const successState = document.getElementById('successState');
        const identityName = document.getElementById('identityName');
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceText = document.getElementById('confidenceText');
        
        identityName.textContent = result.identity.toUpperCase();
        confidenceFill.style.width = `${result.confidence}%`;
        confidenceText.textContent = `${result.confidence.toFixed(1)}% de confianza`;
        
        successState.style.display = 'flex';
        
        // Show probabilities
        this.showProbabilities(result.probabilities);
        
        this.showNotification(`¡Bienvenido, ${result.identity}!`, 'success');
    }
    
    showUnknown(result) {
        const unknownState = document.getElementById('unknownState');
        const unknownMessage = document.getElementById('unknownMessage');
        
        unknownMessage.textContent = result.message;
        unknownState.style.display = 'flex';
        
        // Show probabilities
        this.showProbabilities(result.probabilities);
        
        this.showNotification('Persona no reconocida', 'warning');
    }
    
    showError(title, message) {
        const errorState = document.getElementById('errorState');
        const errorTitle = errorState.querySelector('.error-title');
        const errorMessage = document.getElementById('errorMessage');
        
        errorTitle.textContent = title;
        errorMessage.textContent = message;
        errorState.style.display = 'flex';
        
        this.showNotification(title, 'error');
    }
    
    showProbabilities(probabilities) {
        if (!probabilities || Object.keys(probabilities).length === 0) {
            return;
        }
        
        const section = document.getElementById('probabilitiesSection');
        const list = document.getElementById('probabilitiesList');
        
        // Clear previous
        list.innerHTML = '';
        
        // Sort by probability descending
        const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
        
        // Create bars
        sorted.forEach(([name, prob]) => {
            const item = document.createElement('div');
            item.className = 'probability-item';
            
            const label = document.createElement('span');
            label.className = 'probability-label';
            label.textContent = name;
            
            const barContainer = document.createElement('div');
            barContainer.className = 'probability-bar';
            
            const bar = document.createElement('div');
            bar.className = 'probability-fill';
            bar.style.width = `${prob}%`;
            
            const value = document.createElement('span');
            value.className = 'probability-value';
            value.textContent = `${prob.toFixed(1)}%`;
            
            barContainer.appendChild(bar);
            item.appendChild(label);
            item.appendChild(barContainer);
            item.appendChild(value);
            list.appendChild(item);
        });
        
        section.style.display = 'block';
    }
    
    hideAllStates() {
        document.getElementById('resultCard').style.display = 'none';
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('successState').style.display = 'none';
        document.getElementById('unknownState').style.display = 'none';
        document.getElementById('errorState').style.display = 'none';
        document.getElementById('probabilitiesSection').style.display = 'none';
    }
    
    showNotification(message, type = 'info') {
        // Simple console notification (you can implement a toast later)
        const emoji = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        };
        
        console.log(`${emoji[type]} ${message}`);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new FaceAuthApp();
    console.log('✅ Face Authentication App initialized');
});

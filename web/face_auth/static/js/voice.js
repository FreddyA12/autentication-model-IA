/**
 * Voice Authentication Handler
 */

let audioRecorder = null;
let recordedBlob = null;
let visualizerContext = null;
let visualizerAnalyser = null;
let animationId = null;

const startRecordingBtn = document.getElementById('startRecording');
const stopRecordingBtn = document.getElementById('stopRecording');
const authenticateVoiceBtn = document.getElementById('authenticateVoice');
const recordingStatus = document.getElementById('recordingStatus');
const micIcon = document.getElementById('micIcon');
const audioPlayer = document.getElementById('audioPlayer');
const recordedAudio = document.getElementById('recordedAudio');
const waveformCanvas = document.getElementById('waveform');
const waveformCtx = waveformCanvas.getContext('2d');

// Result elements
const voiceResultCard = document.getElementById('voiceResultCard');
const voiceLoadingState = document.getElementById('voiceLoadingState');
const voiceSuccessState = document.getElementById('voiceSuccessState');
const voiceUnknownState = document.getElementById('voiceUnknownState');
const voiceErrorState = document.getElementById('voiceErrorState');
const voiceProbabilitiesSection = document.getElementById('voiceProbabilitiesSection');

/**
 * Start recording audio
 */
startRecordingBtn.addEventListener('click', async function() {
    try {
        recordedBlob = null;
        
        // Crear nuevo recorder
        audioRecorder = new AudioRecorder();
        await audioRecorder.start();
        
        console.log('🎙️ Grabación iniciada');
        
        // Setup audio visualization
        visualizerContext = audioRecorder.getAudioContext();
        const source = visualizerContext.createMediaStreamSource(audioRecorder.mediaStream);
        visualizerAnalyser = visualizerContext.createAnalyser();
        visualizerAnalyser.fftSize = 2048;
        source.connect(visualizerAnalyser);
        
        // Update UI
        startRecordingBtn.style.display = 'none';
        stopRecordingBtn.style.display = 'flex';
        stopRecordingBtn.disabled = false;
        recordingStatus.classList.add('recording');
        recordingStatus.querySelector('.status-text').textContent = 'Grabando...';
        micIcon.classList.add('recording');
        
        // Start visualization
        visualizeAudio();
        
        // Auto-stop after 3 seconds
        setTimeout(() => {
            if (audioRecorder && audioRecorder.isRecording) {
                stopRecordingBtn.click();
            }
        }, 3000);
        
    } catch (error) {
        console.error('Error al iniciar grabación:', error);
        showVoiceError('No se pudo acceder al micrófono: ' + error.message);
    }
});

/**
 * Stop recording
 */
stopRecordingBtn.addEventListener('click', function() {
    if (audioRecorder && audioRecorder.isRecording) {
        audioRecorder.stop();
        
        // Obtener el WAV
        recordedBlob = audioRecorder.getWAVBlob();
        
        console.log('🎵 Audio grabado como WAV:', {
            size: recordedBlob.size,
            type: recordedBlob.type
        });
        
        // Crear URL para reproducción
        const audioUrl = URL.createObjectURL(recordedBlob);
        recordedAudio.src = audioUrl;
        audioPlayer.style.display = 'block';
        authenticateVoiceBtn.disabled = false;
        
        // Stop visualization
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
        
        // Update UI
        startRecordingBtn.style.display = 'flex';
        stopRecordingBtn.style.display = 'none';
        recordingStatus.classList.remove('recording');
        recordingStatus.querySelector('.status-text').textContent = 'Grabación completada';
        micIcon.classList.remove('recording');
    }
});

/**
 * Visualize audio waveform
 */
function visualizeAudio() {
    if (!visualizerAnalyser) return;
    
    const bufferLength = visualizerAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        animationId = requestAnimationFrame(draw);
        
        visualizerAnalyser.getByteTimeDomainData(dataArray);
        
        waveformCtx.fillStyle = 'rgba(15, 22, 41, 0.3)';
        waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        
        waveformCtx.lineWidth = 2;
        waveformCtx.strokeStyle = '#22d3ee';
        waveformCtx.beginPath();
        
        const sliceWidth = waveformCanvas.width / bufferLength;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * waveformCanvas.height) / 2;
            
            if (i === 0) {
                waveformCtx.moveTo(x, y);
            } else {
                waveformCtx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        waveformCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
        waveformCtx.stroke();
    }
    
    draw();
}

/**
 * Authenticate voice
 */
authenticateVoiceBtn.addEventListener('click', async function() {
    if (!recordedBlob) {
        showVoiceError('No hay audio grabado');
        return;
    }
    
    // Show loading
    hideAllVoiceStates();
    voiceLoadingState.style.display = 'flex';
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('audio', recordedBlob, 'recording.wav');
        
        console.log('🎤 Enviando audio para autenticación...');
        
        // Send to API
        const response = await fetch('/api/predict_voice/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        console.log('✅ Respuesta:', result);
        
        if (result.success) {
            if (result.identity === 'unknown') {
                showVoiceUnknown(result);
            } else {
                showVoiceSuccess(result);
            }
        } else {
            showVoiceError(result.error || result.message || 'Error desconocido');
        }
        
    } catch (error) {
        console.error('❌ Error:', error);
        showVoiceError('Error de conexión: ' + error.message);
    }
});

/**
 * Show success state
 */
function showVoiceSuccess(result) {
    hideAllVoiceStates();
    voiceSuccessState.style.display = 'flex';
    
    document.getElementById('voiceIdentityName').textContent = result.identity.toUpperCase();
    document.getElementById('voiceConfidenceFill').style.width = (result.confidence * 100) + '%';
    document.getElementById('voiceConfidenceText').textContent = (result.confidence * 100).toFixed(1) + '%';
    
    // Show probabilities
    if (result.probabilities) {
        showVoiceProbabilities(result.probabilities);
    }
}

/**
 * Show unknown state
 */
function showVoiceUnknown(result) {
    hideAllVoiceStates();
    voiceUnknownState.style.display = 'flex';
    
    const confidence = (result.confidence * 100).toFixed(1);
    document.getElementById('voiceUnknownMessage').textContent = 
        `No se pudo identificar la voz (confianza máxima: ${confidence}%)`;
    
    // Show probabilities
    if (result.probabilities) {
        showVoiceProbabilities(result.probabilities);
    }
}

/**
 * Show error state
 */
function showVoiceError(message) {
    hideAllVoiceStates();
    voiceErrorState.style.display = 'flex';
    document.getElementById('voiceErrorMessage').textContent = message;
}

/**
 * Show probabilities
 */
function showVoiceProbabilities(probabilities) {
    const list = document.getElementById('voiceProbabilitiesList');
    list.innerHTML = '';
    
    // Sort by probability descending
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    
    sorted.forEach(([name, prob]) => {
        const item = document.createElement('div');
        item.className = 'probability-item';
        
        const percentage = (prob * 100).toFixed(1);
        
        item.innerHTML = `
            <span class="probability-label">${name}</span>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${percentage}%"></div>
            </div>
            <span class="probability-value">${percentage}%</span>
        `;
        
        list.appendChild(item);
    });
    
    voiceProbabilitiesSection.style.display = 'block';
}

/**
 * Hide all result states
 */
function hideAllVoiceStates() {
    voiceResultCard.style.display = 'none';
    voiceLoadingState.style.display = 'none';
    voiceSuccessState.style.display = 'none';
    voiceUnknownState.style.display = 'none';
    voiceErrorState.style.display = 'none';
    voiceProbabilitiesSection.style.display = 'none';
}

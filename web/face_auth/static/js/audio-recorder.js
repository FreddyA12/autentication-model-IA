/**
 * Audio Recorder con conversión a WAV
 * Captura audio y lo convierte a WAV en el navegador
 */

class AudioRecorder {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.audioBuffers = [];
        this.isRecording = false;
        this.sampleRate = 16000;
    }
    
    async start() {
        try {
            // Solicitar acceso al micrófono
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: this.sampleRate
                }
            });
            
            // Crear contexto de audio
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
            
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            // Crear procesador con ScriptProcessorNode (deprecated pero funcional)
            const bufferSize = 4096;
            this.processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            this.audioBuffers = [];
            this.isRecording = true;
            
            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                
                const inputData = e.inputBuffer.getChannelData(0);
                // Copiar datos
                this.audioBuffers.push(new Float32Array(inputData));
            };
            
            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            return true;
        } catch (error) {
            console.error('Error al iniciar grabación:', error);
            throw error;
        }
    }
    
    stop() {
        this.isRecording = false;
        
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
    
    getWAVBlob() {
        // Concatenar todos los buffers
        const totalLength = this.audioBuffers.reduce((acc, buf) => acc + buf.length, 0);
        const audioData = new Float32Array(totalLength);
        
        let offset = 0;
        for (const buffer of this.audioBuffers) {
            audioData.set(buffer, offset);
            offset += buffer.length;
        }
        
        // Convertir a WAV
        const wavBuffer = this.encodeWAV(audioData, this.sampleRate);
        return new Blob([wavBuffer], { type: 'audio/wav' });
    }
    
    encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        
        // Escribir encabezado WAV
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        // RIFF identifier
        writeString(0, 'RIFF');
        // File length
        view.setUint32(4, 36 + samples.length * 2, true);
        // RIFF type
        writeString(8, 'WAVE');
        // Format chunk identifier
        writeString(12, 'fmt ');
        // Format chunk length
        view.setUint32(16, 16, true);
        // Sample format (1 = PCM)
        view.setUint16(20, 1, true);
        // Channel count (1 = mono)
        view.setUint16(22, 1, true);
        // Sample rate
        view.setUint32(24, sampleRate, true);
        // Byte rate (sample rate * block align)
        view.setUint32(28, sampleRate * 2, true);
        // Block align (channel count * bytes per sample)
        view.setUint16(32, 2, true);
        // Bits per sample
        view.setUint16(34, 16, true);
        // Data chunk identifier
        writeString(36, 'data');
        // Data chunk length
        view.setUint32(40, samples.length * 2, true);
        
        // Escribir datos PCM
        const volume = 0.8;
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            let sample = samples[i] * volume;
            // Clamp
            sample = Math.max(-1, Math.min(1, sample));
            // Convertir a 16-bit PCM
            const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset, int16, true);
            offset += 2;
        }
        
        return buffer;
    }
    
    getAudioContext() {
        return this.audioContext;
    }
}

// Exportar
window.AudioRecorder = AudioRecorder;

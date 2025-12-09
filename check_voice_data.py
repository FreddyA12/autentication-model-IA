import numpy as np
from pathlib import Path

# Cargar embeddings
data = np.load('dataset/models/voice_embeddings.npz')
embeddings = data['embeddings']
labels = data['labels']

print(f"📊 Datos de entrenamiento de voz:")
print(f"   Total embeddings: {embeddings.shape}")
print(f"   Dimensión: {embeddings.shape[1]}D")
print(f"   Total labels: {len(labels)}")
print(f"   Labels únicos: {np.unique(labels)}")
print(f"\n📈 Distribución por persona:")

unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"   Clase {label}: {count} muestras")

print(f"\n⚠️  Problema detectado:")
if len(labels) < 50:
    print(f"   SOLO {len(labels)} muestras totales - MUY POCAS para entrenar bien!")
    print(f"   Recomendado: Al menos 100+ muestras por persona")
    print(f"   Solución: Generar más audios con dataset/scripts/5_extract_audio.py")

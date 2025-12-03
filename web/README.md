# 🌐 Frontend Web Django - Sistema de Reconocimiento Facial

Aplicación web moderna para autenticación facial en tiempo real usando Django.

---

## 📋 Requisitos

```bash
# Instalar Django y dependencias
pip install django pillow
```

Las dependencias de reconocimiento facial ya están instaladas del entrenamiento:
- tensorflow
- torch
- facenet-pytorch
- opencv-python

---

## 🚀 Iniciar el Servidor

### Paso 1: Migrar la base de datos

```powershell
cd web
python manage.py migrate
```

### Paso 2: Ejecutar el servidor

```powershell
python manage.py runserver
```

### Paso 3: Abrir en el navegador

```
http://localhost:8000
```

---

## 🎨 Características

### Frontend Moderno
- **Interfaz oscura** con gradientes y animaciones
- **Cámara web en vivo** con guías visuales
- **Resultados en tiempo real**:
  - ✅ Persona autenticada (verde)
  - ⚠️ Persona desconocida (amarillo)
  - ❌ Error o sin rostro detectado (rojo)
- **Gráficas de probabilidad** para cada persona
- **Responsive design** (funciona en móviles)

### Backend Robusto
- **API REST** en `/api/predict/`
- **FaceNet + CNN** para reconocimiento
- **Detección MTCNN** automática
- **Umbral configurable** (50% por defecto)

---

## 📁 Estructura del Proyecto

```
web/
├── manage.py                    # Django manager
├── settings.py                  # Configuración Django
├── urls.py                      # URLs principales
├── wsgi.py / asgi.py           # Servidores
│
├── face_auth/                   # App principal
│   ├── views.py                 # Lógica de vistas
│   ├── urls.py                  # URLs de la app
│   ├── face_service.py          # Servicio de reconocimiento
│   │
│   ├── templates/
│   │   └── index.html           # Página principal
│   │
│   └── static/
│       ├── css/
│       │   └── style.css        # Estilos modernos
│       └── js/
│           └── camera.js        # Lógica de cámara
│
└── db.sqlite3                   # Base de datos (auto-generada)
```

---

## 🔧 Configuración

### Archivo: `web/settings.py`

```python
# Rutas de los modelos
FACE_MODEL_PATH = 'dataset/models/face_classifier.keras'
CLASS_INDICES_PATH = 'dataset/models/class_indices.json'

# Umbral de confianza (50%)
CONFIDENCE_THRESHOLD = 0.50
```

**Para ajustar la sensibilidad:**
- **Más estricto** → `CONFIDENCE_THRESHOLD = 0.70` (menos falsos positivos)
- **Menos estricto** → `CONFIDENCE_THRESHOLD = 0.40` (menos falsos negativos)

---

## 🎯 Cómo Usar

1. **Abrir navegador** en `http://localhost:8000`
2. **Permitir acceso a la cámara** cuando el navegador lo solicite
3. **Activar cámara** con el botón azul
4. **Posicionar rostro** dentro de las guías visuales
5. **Capturar imagen** con el botón verde
6. **Ver resultado** instantáneo con probabilidades

---

## 📡 API Endpoints

### `POST /api/predict/`

Predice la identidad de un rostro en una imagen.

**Request:**
```http
POST /api/predict/
Content-Type: multipart/form-data

image: [archivo de imagen]
```

**Response (Éxito):**
```json
{
  "success": true,
  "identity": "freddy",
  "confidence": 98.5,
  "probabilities": {
    "freddy": 98.5,
    "alison": 1.2,
    "isma": 0.3
  },
  "message": "Rostro reconocido como freddy"
}
```

**Response (Desconocido):**
```json
{
  "success": false,
  "identity": "DESCONOCIDO",
  "confidence": 43.8,
  "probabilities": {
    "alison": 43.8,
    "isma": 40.6,
    "freddy": 15.6
  },
  "message": "Confianza insuficiente (max: 43.8%)"
}
```

**Response (Sin rostro):**
```json
{
  "success": false,
  "identity": null,
  "confidence": 0,
  "probabilities": {},
  "message": "No se detectó ningún rostro en la imagen"
}
```

---

## 🧪 Pruebas con cURL

```powershell
# Probar con una imagen
curl -X POST -F "image=@C:\ruta\a\imagen.jpg" http://localhost:8000/api/predict/
```

---

## 🎨 Personalización del Frontend

### Cambiar colores (en `static/css/style.css`)

```css
:root {
    --primary-color: #3b82f6;      /* Color principal */
    --success-color: #10b981;      /* Color de éxito */
    --warning-color: #f59e0b;      /* Color de advertencia */
    --error-color: #ef4444;        /* Color de error */
}
```

### Cambiar textos (en `templates/index.html`)

Busca y modifica los textos HTML directamente.

---

## 🔐 Seguridad

### Para producción:

1. **Cambiar SECRET_KEY** en `settings.py`:
```python
SECRET_KEY = 'tu-clave-secreta-segura-aqui'
```

2. **Desactivar DEBUG**:
```python
DEBUG = False
```

3. **Configurar ALLOWED_HOSTS**:
```python
ALLOWED_HOSTS = ['tu-dominio.com', 'www.tu-dominio.com']
```

4. **Usar HTTPS** con certificado SSL

5. **Habilitar CSRF** (ya está activado por defecto)

---

## 🐛 Solución de Problemas

### Error: "No se pudo acceder a la cámara"
- **Chrome/Edge:** Verifica que el sitio tenga permisos de cámara
- **HTTPS:** En producción, la cámara requiere HTTPS

### Error: "No module named 'face_auth'"
```powershell
# Asegúrate de estar en la carpeta web/
cd web
python manage.py runserver
```

### Error: "No se detectó ningún rostro"
- Asegúrate de que haya buena iluminación
- Posiciona tu rostro dentro de las guías
- El rostro debe ocupar al menos el 30% de la imagen

### Los modelos no se cargan
```powershell
# Verifica que existan los modelos
ls ..\dataset\models\face_classifier.keras
ls ..\dataset\models\class_indices.json
```

---

## 📊 Rendimiento

- **Detección MTCNN:** ~100-200ms
- **Extracción FaceNet:** ~50-100ms
- **Clasificación CNN:** ~10-20ms
- **Total:** ~200-400ms por imagen

**Optimización:**
- Usar GPU si está disponible (detectado automáticamente)
- Reducir resolución de video (ya configurado en 1280x720)

---

## 📝 Próximos Pasos

### Funcionalidades adicionales:

1. **Dashboard de estadísticas**
   - Conteo de autenticaciones
   - Historial de accesos
   - Gráficas de uso

2. **Múltiples cámaras**
   - Seleccionar cámara delantera/trasera
   - Soporte para múltiples dispositivos

3. **Modo foto**
   - Subir imagen desde el disco
   - Probar con fotos guardadas

4. **Autenticación dual**
   - Integrar reconocimiento de voz
   - Requiere ambos para autenticar

5. **Base de datos**
   - Guardar logs de autenticación
   - Registro de usuarios

---

## 📞 Soporte

Si tienes problemas:
1. Revisa la consola del navegador (F12)
2. Revisa los logs del servidor Django
3. Verifica que los modelos estén correctamente entrenados

---

## ✅ Checklist de Instalación

- [ ] Django instalado (`pip install django pillow`)
- [ ] Modelos entrenados en `dataset/models/`
- [ ] Migración ejecutada (`python manage.py migrate`)
- [ ] Servidor corriendo (`python manage.py runserver`)
- [ ] Navegador en `http://localhost:8000`
- [ ] Permisos de cámara concedidos
- [ ] Probado con tu rostro ✅

---

¡Listo! Tu sistema de reconocimiento facial con Django está funcionando 🚀

import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)

# Dataset paths
AUDIO_DATASET_DIR  = os.path.join(ROOT_DIR, 'AUDIO_DATASET')
IMAGES_DATASET_DIR = os.path.join(ROOT_DIR, 'IMAGES_DATASET')

# Model & results paths
MODELS_DIR       = os.path.join(BASE_DIR, 'saved_models')
AUDIO_MODEL_PATH       = os.path.join(MODELS_DIR, 'audio_classifier.pth')
AUDIO_MODEL_BOOST_PATH = os.path.join(MODELS_DIR, 'audio_classifier_boost.pth')
YOLO_CUSTOM_PATH     = os.path.join(MODELS_DIR, 'yolo_custom_cls.pt')
YOLO_CUSTOM_DET_PATH = os.path.join(MODELS_DIR, 'yolo_custom_det.pt')
RESULTS_DIR      = os.path.join(BASE_DIR, 'results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Animals (7 classes in dataset, lowercase)
ANIMAL_CLASSES = ['bear', 'cat', 'cow', 'elephant', 'horse', 'lion', 'sheep']

# YOLO COCO class IDs → animal name (subset that matches our dataset)
YOLO_COCO_ANIMALS = {
    15: 'cat',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
}

# Email (set via /email/configure endpoint or env vars)
SMTP_HOST    = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT    = int(os.getenv('SMTP_PORT', '587'))
EMAIL_SENDER = os.getenv('EMAIL_SENDER', '')
EMAIL_PASS   = os.getenv('EMAIL_PASS', '')

# Ollama
OLLAMA_HOST  = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')

# Audio feature params
AUDIO_SR       = 22050
AUDIO_DURATION = 4.0     # seconds
N_MFCC         = 40      # MFCC coefficients
N_MELS         = 128     # Mel-spectrogram bands
N_FFT          = 2048    # FFT window size
HOP_LENGTH     = 512     # hop between frames
N_CHANNELS     = 2       # input channels: [mel-spectrogram, MFCC]

# Flask server
# 5001 is sometimes taken by the Samsung "GT3 mobile service" on this PC —
# using 5055 to avoid the conflict. Overridable via env for server deploys.
HOST = os.getenv('WILDLIFE_HOST', '127.0.0.1')
PORT = int(os.getenv('WILDLIFE_PORT', '5055'))

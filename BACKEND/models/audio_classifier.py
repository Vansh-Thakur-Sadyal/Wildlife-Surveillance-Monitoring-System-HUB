"""
Audio classifier — 2-channel spectrogram input (Mel + MFCC).
Supports two model architectures:
  AudioCNN        — original CNN (from first training run)
  SEResAudioCNN   — improved: SE-ResNet with channel attention (boost model)

Load priority: audio_classifier_boost.pth > audio_classifier.pth
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (AUDIO_MODEL_PATH, AUDIO_MODEL_BOOST_PATH, ANIMAL_CLASSES,
                    AUDIO_SR, AUDIO_DURATION, N_MFCC,
                    N_MELS, N_FFT, HOP_LENGTH, N_CHANNELS)

try:
    import librosa
    import cv2
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    print("[Audio] librosa/cv2 not installed.")

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[Audio] PyTorch not installed.")


# ═════════════════════════════════════════════════════════════════════════════
#  Architecture 1 — original AudioCNN  (used by first training run)
# ═════════════════════════════════════════════════════════════════════════════
class AudioCNN(nn.Module):
    def __init__(self, n_classes, n_channels=N_CHANNELS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(0.20),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(0.30),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.50),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.35),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ═════════════════════════════════════════════════════════════════════════════
#  Architecture 2 — SEResAudioCNN  (boost model — better for weak classes)
# ═════════════════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns which frequency channels matter most."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """Residual block + SE channel attention."""
    def __init__(self, ch, dropout=0.20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.se  = SEBlock(ch)
        self.act = nn.ReLU(inplace=True)
        self.drop= nn.Dropout2d(dropout)

    def forward(self, x):
        return self.act(x + self.drop(self.se(self.conv(x))))


class SEResAudioCNN(nn.Module):
    """
    SE-ResNet for audio spectrogram classification.
    Deeper than AudioCNN + residual connections + channel attention.
    Much better at distinguishing similar-sounding classes (cat/lion/sheep).
    """
    def __init__(self, n_classes, n_channels=N_CHANNELS):
        super().__init__()
        # Stem: large kernel to capture broad spectral patterns first
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # Stage 1: 64 channels
        self.stage1 = nn.Sequential(
            ResBlock(64, dropout=0.15), ResBlock(64, dropout=0.15),
            nn.MaxPool2d(2),
        )
        # Stage 2: 128 channels
        self.up2    = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False),
                                    nn.BatchNorm2d(128))
        self.stage2 = nn.Sequential(
            ResBlock(128, dropout=0.20), ResBlock(128, dropout=0.20),
            nn.MaxPool2d(2),
        )
        # Stage 3: 256 channels
        self.up3    = nn.Sequential(nn.Conv2d(128, 256, 1, bias=False),
                                    nn.BatchNorm2d(256))
        self.stage3 = nn.Sequential(
            ResBlock(256, dropout=0.25), ResBlock(256, dropout=0.25),
        )
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.BatchNorm1d(512),
            nn.GELU(), nn.Dropout(0.50),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(self.up2(x))
        x = self.stage3(self.up3(x))
        return self.head(x)


# ═════════════════════════════════════════════════════════════════════════════
#  Feature extraction — shared by training & inference
# ═════════════════════════════════════════════════════════════════════════════
def _normalize(arr):
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def extract_dual_features(audio_path, to_wav_fn=None):
    """
    Returns np.ndarray shape (2, N_MELS, T) — float32.
      ch0: log mel-spectrogram (normalized)
      ch1: MFCC resized to N_MELS rows (normalized)
    """
    if not LIBROSA_OK:
        return None
    wav_path, is_tmp = (to_wav_fn(audio_path) if to_wav_fn
                        else (audio_path, False))
    try:
        y, sr = librosa.load(wav_path, sr=AUDIO_SR,
                             duration=AUDIO_DURATION, mono=True)
        target = int(AUDIO_SR * AUDIO_DURATION)
        y = np.pad(y, (0, max(0, target - len(y))))[:target]

        # ch0: Log Mel-Spectrogram
        mel    = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_db = _normalize(librosa.power_to_db(mel, ref=np.max)).astype(np.float32)
        T      = mel_db.shape[1]

        # ch1: MFCC resized to (N_MELS, T)
        mfcc = _normalize(librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=N_MFCC,
            n_fft=N_FFT, hop_length=HOP_LENGTH)).astype(np.float32)
        mfcc_r = cv2.resize(mfcc, (T, N_MELS), interpolation=cv2.INTER_LINEAR)

        return np.stack([mel_db, mfcc_r], axis=0)   # (2, 128, T)
    except Exception as e:
        print(f"[Audio] Feature error: {e}")
        return None
    finally:
        if is_tmp and os.path.exists(wav_path):
            os.unlink(wav_path)


# ═════════════════════════════════════════════════════════════════════════════
#  AudioClassifier — loads boost model if available, else original
# ═════════════════════════════════════════════════════════════════════════════
class AudioClassifier:
    def __init__(self):
        self.model   = None
        self.device  = None
        self.classes = ANIMAL_CLASSES
        self._load_model()

    def _load_model(self):
        if not TORCH_OK:
            return
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prefer boost model (better accuracy on weak classes)
        for path, arch_name in [(AUDIO_MODEL_BOOST_PATH, 'SEResAudioCNN'),
                                 (AUDIO_MODEL_PATH,       'AudioCNN')]:
            if not os.path.exists(path):
                continue
            try:
                ckpt    = torch.load(path, map_location=self.device,
                                     weights_only=False)
                n_ch    = ckpt.get('n_channels', N_CHANNELS)
                arch    = ckpt.get('arch', arch_name)
                Model   = SEResAudioCNN if arch == 'SEResAudioCNN' else AudioCNN
                self.model = Model(n_classes=len(self.classes),
                                   n_channels=n_ch).to(self.device)
                self.model.load_state_dict(ckpt['model_state'])
                self.model.eval()
                tag = '(BOOST)' if 'boost' in path else ''
                # ASCII only — a Unicode arrow here crashes under cp1252
                # consoles, which aborts the load mid-way
                print(f"[Audio] {tag} {arch} loaded ({self.device}) <- {path}")
                return
            except Exception as e:
                print(f"[Audio] Load error ({path}): {e}")

        print("[Audio] No model found. Run train_audio.bat or train_boost.bat")

    def _to_wav(self, path):
        if path.lower().endswith('.wav'):
            return path, False
        if not PYDUB_OK:
            return path, False
        try:
            seg = AudioSegment.from_file(path)
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp.close()
            seg.export(tmp.name, format='wav')
            return tmp.name, True
        except Exception as e:
            print(f"[Audio] Conversion error: {e}")
            return path, False

    def extract_features(self, audio_path):
        return extract_dual_features(audio_path, to_wav_fn=self._to_wav)

    def rms_level(self, audio_path):
        """Mean RMS loudness of the clip (0..1). Used to skip silent chunks."""
        info = self.activity_level(audio_path)
        return info['rms'] if info else None

    def activity_level(self, audio_path):
        """
        Loudness + dynamics of a clip.
          rms      — overall level (0..1)
          dynamics — p95/median of frame RMS. Steady noise (fan, hum) ≈ 1–1.5;
                     real animal sounds burst + pause, giving much higher values.
        """
        if not LIBROSA_OK:
            return None
        wav, is_tmp = self._to_wav(audio_path)
        try:
            y, _ = librosa.load(wav, sr=AUDIO_SR, mono=True, duration=AUDIO_DURATION)
            if not y.size:
                return {'rms': 0.0, 'dynamics': 0.0}
            frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            med = float(np.median(frames))
            p95 = float(np.percentile(frames, 95))
            return {'rms': float(np.sqrt(np.mean(y ** 2))),
                    'dynamics': p95 / (med + 1e-8)}
        except Exception:
            return None
        finally:
            if is_tmp and os.path.exists(wav):
                os.unlink(wav)

    def predict(self, audio_path):
        if self.model is None:
            return {'animal': 'unknown', 'confidence': 0.0,
                    'error': 'No model loaded. Run train_audio.bat first.'}
        feats = self.extract_features(audio_path)
        if feats is None:
            return {'animal': 'unknown', 'confidence': 0.0,
                    'error': 'Feature extraction failed.'}
        x = torch.from_numpy(feats[None]).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return {
            'animal':     self.classes[idx],
            'confidence': round(float(min(probs[idx], 0.999)), 4),
            'all_probs':  {c: round(float(p), 4)
                           for c, p in zip(self.classes, probs)},
        }

"""
YOLODetector — model priority at startup:

1. yolo_custom_det.pt (OUR trained detector, from train_detect.bat)
   → real boxes + labels, all from our own model. Nothing pretrained runs.

2. yolo_custom_cls.pt with CUSTOM_ONLY=True (our trained classifier)
   → whole-image classification, bbox spans the full frame.

3. Legacy two-stage (CUSTOM_ONLY=False, no custom detector):
   yolov8n.pt COCO boxes + custom classifier label refinement.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (YOLO_CUSTOM_PATH, YOLO_CUSTOM_DET_PATH,
                    YOLO_COCO_ANIMALS, ANIMAL_CLASSES)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[YOLO] ultralytics not installed. Run: pip install ultralytics")


BBOX_COLORS = {
    'elephant': (0, 245, 255),  'zebra':  (0, 255, 136),
    'buffalo':  (180, 76, 255), 'rhino':  (255, 136, 0),
    'bear':     (100, 160, 255),'cat':    (100, 255, 100),
    'cow':      (100, 100, 255),'horse':  (255, 200, 0),
    'lion':     (255, 165, 0),  'sheep':  (200, 200, 200),
}

DETECTION_CONF = 0.30   # confidence threshold — raise to reduce false positives
DETECTION_IOU  = 0.45   # NMS IoU threshold — lower = more aggressive duplicate removal
CLS_CONF_MIN   = 0.40   # minimum confidence to use custom classifier result

CUSTOM_ONLY    = True   # True → only our trained classifier runs; COCO model unused


class YOLODetector:
    def __init__(self):
        self.det_model  = None   # detection model (custom preferred, COCO legacy)
        self.cls_model  = None   # YOLOv8 classification (custom)
        self.custom_det = False  # True → det_model is OUR trained detector
        self._load_models()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_models(self):
        if not YOLO_AVAILABLE:
            return

        # ── OUR detection model (best option: boxes + labels, all ours) ───
        if os.path.exists(YOLO_CUSTOM_DET_PATH):
            try:
                self.det_model  = YOLO(YOLO_CUSTOM_DET_PATH)
                self.custom_det = True
                print(f"[YOLO] CUSTOM detector: {YOLO_CUSTOM_DET_PATH} (our model, boxes + labels)")
                return
            except Exception as e:
                print(f"[YOLO] Custom detector load failed: {e}")

        # ── Classification model (custom trained) ─────────────────────────
        if os.path.exists(YOLO_CUSTOM_PATH):
            try:
                self.cls_model = YOLO(YOLO_CUSTOM_PATH)
                print(f"[YOLO] Custom classifier: {YOLO_CUSTOM_PATH}")
            except Exception as e:
                print(f"[YOLO] Custom classifier load failed: {e}")
        else:
            print("[YOLO] No custom classifier yet — run train_yolo.bat")

        if CUSTOM_ONLY and self.cls_model is not None:
            print("[YOLO] CUSTOM_ONLY mode — pretrained COCO model disabled")
            return

        # ── Detection model (pretrained COCO, legacy / fallback) ──────────
        try:
            self.det_model = YOLO('yolov8n.pt')
            print("[YOLO] Detection model: yolov8n.pt (COCO pretrained)")
        except Exception as e:
            print(f"[YOLO] Detection model load failed: {e}")

    # ── COCO class → animal name ──────────────────────────────────────────────
    def _coco_to_animal(self, cls_id):
        return YOLO_COCO_ANIMALS.get(cls_id, None)

    # ── Crop image region for classification ──────────────────────────────────
    def _classify_crop(self, img_bgr, bbox):
        """Run custom classifier on a detected region to refine the label."""
        if self.cls_model is None:
            return None, 0.0
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Add 10% padding around the crop
        h, w = img_bgr.shape[:2]
        pad_x = int((x2 - x1) * 0.10)
        pad_y = int((y2 - y1) * 0.10)
        x1c = max(0, x1 - pad_x); y1c = max(0, y1 - pad_y)
        x2c = min(w, x2 + pad_x); y2c = min(h, y2 + pad_y)
        crop = img_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return None, 0.0
        try:
            res   = self.cls_model(crop, verbose=False)
            probs = res[0].probs
            if probs is None:
                return None, 0.0
            top1_conf = float(probs.top1conf)
            if top1_conf < CLS_CONF_MIN:
                return None, top1_conf
            cls_name = self.cls_model.names[probs.top1].lower()
            return (cls_name if cls_name in ANIMAL_CLASSES else None), top1_conf
        except Exception as e:
            print(f"[YOLO] Classify crop error: {e}")
            return None, 0.0

    # ── Custom-only detection: whole-image classification ────────────────────
    def _detect_custom_only(self, image_input):
        """Classify the whole image with our trained model. bbox = full frame."""
        img = cv2.imread(image_input) if isinstance(image_input, str) else image_input
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
        try:
            res   = self.cls_model(img, verbose=False)
            probs = res[0].probs
            if probs is None:
                return []
            conf = float(probs.top1conf)
            if conf < CLS_CONF_MIN:
                return []
            animal = self.cls_model.names[probs.top1].lower()
            if animal not in ANIMAL_CLASSES:
                return []
            return [{
                'id':         1,
                'animal':     animal,
                'confidence': round(min(conf, 0.999), 4),
                'bbox':       [0.0, 0.0, float(w), float(h)],
            }]
        except Exception as e:
            print(f"[YOLO] Custom classify error: {e}")
            return []

    # ── Detection with OUR trained detector ──────────────────────────────────
    def _detect_with_custom_det(self, image_input, conf_threshold):
        """Boxes AND labels straight from our trained model. Nothing pretrained."""
        try:
            results = self.det_model(image_input, conf=conf_threshold,
                                     iou=DETECTION_IOU, verbose=False)
        except Exception as e:
            print(f"[YOLO] Custom detection error: {e}")
            return []

        detections, det_id = [], 1
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                animal = self.det_model.names[int(box.cls[0])].lower()
                if animal not in ANIMAL_CLASSES:
                    continue
                conf = min(float(box.conf[0]), 0.999)
                detections.append({
                    'id':         det_id,
                    'animal':     animal,
                    'confidence': round(conf, 4),
                    'bbox':       [round(v, 1) for v in box.xyxy[0].tolist()],
                })
                det_id += 1
        return detections

    # ── Core detection ────────────────────────────────────────────────────────
    def detect(self, image_input, conf_threshold=DETECTION_CONF):
        """
        Detect animals. image_input = file path or numpy array.
        Returns list of {id, animal, confidence, bbox}
        """
        if self.custom_det:
            return self._detect_with_custom_det(image_input, conf_threshold)

        if CUSTOM_ONLY and self.cls_model is not None:
            return self._detect_custom_only(image_input)

        if self.det_model is None:
            return []

        try:
            results = self.det_model(image_input, conf=conf_threshold,
                                     iou=DETECTION_IOU, verbose=False)
        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
            return []

        # Load image for crop classification if custom model available
        img_bgr = None
        if self.cls_model is not None and isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)

        detections = []
        det_id = 1

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                bbox   = [round(v, 1) for v in box.xyxy[0].tolist()]

                # Map COCO class → animal name
                animal = self._coco_to_animal(cls_id)
                if animal is None:
                    continue   # skip non-animal COCO classes

                # Refine with custom classifier if available
                if img_bgr is not None:
                    cls_animal, cls_conf = self._classify_crop(img_bgr, bbox)
                    if cls_animal is not None:
                        animal = cls_animal
                        conf   = cls_conf

                # Cap at 0.999 so display never shows "100%"
                conf = min(conf, 0.999)

                detections.append({
                    'id':         det_id,
                    'animal':     animal,
                    'confidence': round(conf, 4),
                    'bbox':       bbox,
                })
                det_id += 1

        return detections

    # ── Detect + draw annotated image ────────────────────────────────────────
    def detect_and_draw(self, image_path, output_path, conf_threshold=DETECTION_CONF):
        img = cv2.imread(image_path)
        if img is None:
            print(f"[YOLO] Cannot read image: {image_path}")
            return [], None

        detections = self.detect(image_path, conf_threshold)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            color = BBOX_COLORS.get(det['animal'], (255, 255, 255))
            conf_display = min(det['confidence'] * 100, 99.9)
            label = f"{det['animal']}  {conf_display:.1f}%"

            # Bounding box (2 passes for glow effect)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1-1, y1-1), (x2+1, y2+1),
                          tuple(min(255, c+80) for c in color), 1)

            # Label background pill
            font      = cv2.FONT_HERSHEY_SIMPLEX
            font_scale= 0.58
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            pill_y = max(y1 - 10, th + 6)
            cv2.rectangle(img,
                          (x1, pill_y - th - 6), (x1 + tw + 10, pill_y + 2),
                          color, -1)
            cv2.putText(img, label, (x1 + 5, pill_y - 2),
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        cv2.imwrite(output_path, img)
        return detections, output_path

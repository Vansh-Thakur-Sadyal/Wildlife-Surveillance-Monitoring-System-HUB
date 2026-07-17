from datetime import datetime
from threading import Lock


class DetectionStore:
    """Thread-safe in-memory store for all detection events."""

    def __init__(self):
        self._lock    = Lock()
        self._records = []   # list of flat dicts

    def add(self, camera_detections=None, source='image',
            filename=None, audio_result=None, fusion_result=None):
        ts = datetime.now().isoformat()
        with self._lock:
            for d in (camera_detections or []):
                self._records.append({
                    'timestamp':  ts,
                    'source':     source,
                    'filename':   filename or '',
                    'type':       'camera',
                    'animal':     d.get('animal', ''),
                    'confidence': d.get('confidence', 0.0),
                    'bbox':       str(d.get('bbox', [])),
                })
            if audio_result and audio_result.get('animal') not in (None, '', 'unknown'):
                self._records.append({
                    'timestamp':  ts,
                    'source':     source,
                    'filename':   filename or '',
                    'type':       'audio',
                    'animal':     audio_result.get('animal', ''),
                    'confidence': audio_result.get('confidence', 0.0),
                    'bbox':       '',
                })
            if fusion_result and fusion_result.get('primary_animal'):
                self._records.append({
                    'timestamp':  ts,
                    'source':     'fusion',
                    'filename':   filename or '',
                    'type':       'fusion',
                    'animal':     fusion_result.get('primary_animal', ''),
                    'confidence': fusion_result.get('fusion_confidence', 0.0),
                    'bbox':       fusion_result.get('method', ''),
                })

    def get_stats(self):
        """Return {animal: count} for analytics chart."""
        with self._lock:
            stats = {}
            for r in self._records:
                a = r['animal']
                if a:
                    stats[a] = stats.get(a, 0) + 1
        # Always include the 4 core species so the chart never errors
        for a in ['buffalo', 'elephant', 'rhino', 'zebra']:
            stats.setdefault(a, 0)
        return stats

    def get_all(self):
        with self._lock:
            return list(self._records)

    def to_csv_rows(self):
        header = ['timestamp', 'source', 'type', 'filename',
                  'animal', 'confidence', 'bbox']
        with self._lock:
            rows = [header]
            for r in self._records:
                rows.append([
                    r['timestamp'], r['source'], r['type'],
                    r['filename'],  r['animal'],
                    f"{r['confidence']:.4f}", r['bbox'],
                ])
        return rows


store = DetectionStore()

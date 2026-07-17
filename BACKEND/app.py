"""
Wildlife AI — Flask backend
Port: 5001
"""
import os
import sys
import csv
import uuid
import io

from flask      import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, HOST, PORT
from models.yolo_detector    import YOLODetector
from models.audio_classifier import AudioClassifier
from utils.detection_store   import store

# ── App init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model singletons ──────────────────────────────────────────────────────────
print("[Wildlife AI] Loading models …")
yolo       = YOLODetector()
audio_clf  = AudioClassifier()
print("[Wildlife AI] Ready.")

# ═══════════════════════════════════════════════════════════════════════════════
#  Static file serving
# ═══════════════════════════════════════════════════════════════════════════════
FRONTEND_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'frontend'))


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:filename>')
def frontend_files(filename):
    # Fixed API routes always win over this catch-all
    return send_from_directory(FRONTEND_DIR, filename)


@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)

# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/detect', methods=['POST'])
def detect():
    """Static image upload → YOLO detection with annotated output image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400

    file   = request.files['image']
    suffix = os.path.splitext(file.filename or '.jpg')[1] or '.jpg'
    uid    = uuid.uuid4().hex[:10]
    in_p   = os.path.join(RESULTS_DIR, f'in_{uid}{suffix}')
    out_p  = os.path.join(RESULTS_DIR, f'result_{uid}.jpg')
    file.save(in_p)

    detections, _ = yolo.detect_and_draw(in_p, out_p)
    store.add(camera_detections=detections, source='image', filename=file.filename)

    try: os.unlink(in_p)
    except: pass

    return jsonify({
        'detections': detections,
        'image_url':  f'/results/result_{uid}.jpg',
    })


@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """Live camera frame (blob) → YOLO detection (no saved output)."""
    if 'image' not in request.files:
        return jsonify({'detections': []}), 200

    file = request.files['image']
    uid  = uuid.uuid4().hex[:10]
    p    = os.path.join(RESULTS_DIR, f'frame_{uid}.jpg')
    file.save(p)

    # Live frames are blurrier/noisier than uploads — use a lower threshold
    detections = yolo.detect(p, conf_threshold=0.20)
    if detections:
        store.add(camera_detections=detections, source='live_camera')

    try: os.unlink(p)
    except: pass

    return jsonify({'detections': detections})

# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/audio_detect', methods=['POST'])
def audio_detect():
    """Static audio file upload → CNN classification."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio field'}), 400

    file   = request.files['audio']
    suffix = os.path.splitext(file.filename or '.wav')[1] or '.wav'
    uid    = uuid.uuid4().hex[:10]
    p      = os.path.join(RESULTS_DIR, f'audio_{uid}{suffix}')
    file.save(p)

    result = audio_clf.predict(p)
    store.add(source='audio', filename=file.filename, audio_result=result)

    try: os.unlink(p)
    except: pass

    return jsonify(result)


@app.route('/live_audio', methods=['POST'])
def live_audio():
    """Live microphone chunk (webm/ogg blob) → CNN classification."""
    if 'audio' not in request.files:
        return jsonify({'animal': 'unknown', 'confidence': 0.0}), 200

    file = request.files['audio']
    uid  = uuid.uuid4().hex[:10]
    ext  = os.path.splitext(file.filename or 'chunk.webm')[1] or '.webm'
    p    = os.path.join(RESULTS_DIR, f'live_{uid}{ext}')
    file.save(p)

    # Live gates. The model has no "nothing" class, so background noise gets
    # forced into some animal — these gates keep that off the screen.
    #   silence          → skip entirely
    #   flat chunk       → could be hum OR a continuous animal call: accept
    #                      only with HIGH confidence, never show a "maybe"
    #   bursty chunk     → normal bar, low results shown as tentative guess
    LIVE_MIN_RMS   = 0.004
    LIVE_FLAT_DYN  = 1.15   # steady noise ≈ 1.02; bursty animal sounds ≥ 1.2
    LIVE_FLAT_CONF = 0.60   # bar for flat chunks (continuous calls)
    LIVE_MIN_CONF  = 0.50   # bar for bursty chunks
    LIVE_MIN_GUESS = 0.35   # below this, don't even show a "maybe"

    act = audio_clf.activity_level(p) or {'rms': None, 'dynamics': None}
    rms, dyn = act['rms'], act['dynamics']
    if rms is not None and rms < LIVE_MIN_RMS:
        result = {'animal': 'listening', 'confidence': 0.0}
        print(f"[LiveMic] rms={rms:.4f} -> silence, skipped", flush=True)
    else:
        result = audio_clf.predict(p)
        conf  = result.get('confidence', 0)
        guess = result.get('animal')
        flat  = dyn is not None and dyn < LIVE_FLAT_DYN
        print(f"[LiveMic] rms={rms if rms is not None else -1:.4f} "
              f"dyn={dyn if dyn is not None else -1:.2f} "
              f"-> {guess} {conf:.2f}{' (flat)' if flat else ''}", flush=True)
        if not result.get('error'):
            if flat and conf < LIVE_FLAT_CONF:
                # probably room noise — say nothing, suggest nothing
                result = {'animal': 'listening', 'confidence': conf}
            elif not flat and conf < LIVE_MIN_CONF:
                result = {'animal': 'listening', 'confidence': conf}
                if conf >= LIVE_MIN_GUESS:
                    result['guess'] = guess

    if result.get('confidence', 0) > 0.5 and result.get('animal') != 'unknown':
        store.add(source='live_mic', audio_result=result)

    try: os.unlink(p)
    except: pass

    return jsonify(result)

# ═══════════════════════════════════════════════════════════════════════════════
#  STATS
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/stats', methods=['GET'])
def stats():
    return jsonify(store.get_stats())

# ═══════════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/export_csv', methods=['GET'])
def export_csv():
    rows = store.to_csv_rows()
    buf  = io.StringIO()
    csv.writer(buf).writerows(rows)
    out  = io.BytesIO(buf.getvalue().encode('utf-8'))
    out.seek(0)
    return send_file(out, mimetype='text/csv', as_attachment=True,
                     download_name='wildlife_detections_full.csv')


@app.route('/export_scan_csv', methods=['POST'])
def export_scan_csv():
    data       = request.get_json(silent=True) or {}
    detections = data.get('detections', [])
    filename   = data.get('filename', '')
    timestamp  = data.get('timestamp', '')

    rows = [['#', 'Animal', 'Confidence', 'Bounding Box', 'File', 'Timestamp']]
    for i, d in enumerate(detections, 1):
        rows.append([
            i,
            d.get('animal', ''),
            f"{d.get('confidence', 0):.4f}",
            str(d.get('bbox', [])),
            filename,
            timestamp,
        ])

    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    out = io.BytesIO(buf.getvalue().encode('utf-8'))
    out.seek(0)
    return send_file(out, mimetype='text/csv', as_attachment=True,
                     download_name='wildlife_scan_results.csv')

# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"[Wildlife AI] Server → http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True, threaded=True)

/* ====================================================
   GLASSMORPHISM WILDLIFE HUB V2 — app.js
   Backend: http://127.0.0.1:5055
   Fixes:
   - Live camera: browser captures frames → sends to /detect_frame
     (no backend camera conflict / no freeze)
   - Live canvas: continuous RAF draws video + bbox overlay
   - HiDPI chart rendering via devicePixelRatio scaling
   - Static result image sized to match live camera result
   ==================================================== */

// Same-origin when served by the backend (VPS); localhost when opened as a file
const API = location.protocol === 'file:' ? 'http://127.0.0.1:5055' : location.origin;

// ─── PARTICLES ────────────────────────────────────────
(function spawnParticles() {
  const container = document.getElementById('particles');
  for (let i = 0; i < 30; i++) {
    const p = document.createElement('div');
    p.classList.add('particle');
    const size = Math.random() * 4 + 2;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random() * 100}%;
      animation-duration:${Math.random() * 12 + 8}s;
      animation-delay:${Math.random() * 10}s;
      opacity:${Math.random() * 0.6 + 0.2};
    `;
    container.appendChild(p);
  }
})();

// ─── IMAGE DETECTION ─────────────────────────────────
const imgInput    = document.getElementById('imgInput');
const imgFileName = document.getElementById('imgFileName');

imgInput.addEventListener('change', () => {
  const file = imgInput.files[0];
  if (!file) return;
  const name = file.name.length > 22 ? file.name.substring(0, 20) + '...' : file.name;
  imgFileName.textContent = 'Choose File: ' + name;

  const imgEl       = document.getElementById('staticImgPreview');
  const placeholder = document.getElementById('staticImgPlaceholder');
  const reader      = new FileReader();
  reader.onload = (e) => {
    imgEl.src = e.target.result;
    imgEl.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
  };
  reader.readAsDataURL(file);
});

function detectImage() {
  const file = imgInput.files[0];
  if (!file) { showToast('⚠ Please select an image file'); return; }

  const card        = document.querySelector('.card-image-detect');
  const imgEl       = document.getElementById('staticImgPreview');
  const placeholder = document.getElementById('staticImgPlaceholder');
  const caption     = document.getElementById('staticImgCaption');

  card.classList.add('detecting');
  caption.textContent = 'Detecting...';

  if (!imgEl.src || imgEl.style.display === 'none') {
    const reader = new FileReader();
    reader.onload = (e) => {
      imgEl.src = e.target.result;
      imgEl.style.display = 'block';
      if (placeholder) placeholder.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  const form = new FormData();
  form.append('image', file);

  fetch(`${API}/detect`, { method: 'POST', body: form })
    .then(r => r.json())
    .then(data => {
      card.classList.remove('detecting');

      if (data.image_url) {
        const ts = Date.now();
        imgEl.src = `${API}${data.image_url}?t=${ts}`;
        imgEl.style.display = 'block';
        imgEl.onload = () => { if (placeholder) placeholder.style.display = 'none'; };
        imgEl.onerror = () => {
          const reader = new FileReader();
          reader.onload = (e) => { imgEl.src = e.target.result; };
          reader.readAsDataURL(file);
        };
      }

      if (data.detections && data.detections.length > 0) {
        const summary = {};
        data.detections.forEach(d => {
          if (!summary[d.animal]) summary[d.animal] = { count: 0, maxConf: 0 };
          summary[d.animal].count++;
          summary[d.animal].maxConf = Math.max(summary[d.animal].maxConf, d.confidence);
        });
        const lines = Object.entries(summary).map(([a, v]) =>
          `${a} ×${v.count} (${(v.maxConf * 100).toFixed(1)}%)`
        );
        caption.textContent = lines.join(' | ');

        showToast(`✔ Detected: ${data.detections[0].animal} (${data.detections.length} total)`);
        alertAnimal(data.detections[0].animal);

      } else {
        caption.textContent = 'No animals detected';
        showToast('⚠ No animals detected');
      }
    })
    .catch(err => {
      card.classList.remove('detecting');
      caption.textContent = 'Preview shown (backend offline)';
      showToast('⚠ Backend unreachable — preview only');
      console.error(err);
    });
}

// ─── AUDIO DETECTION ─────────────────────────────────
const audioInput       = document.getElementById('audioInput');
const audioFileName    = document.getElementById('audioFileName');
const audioPlayer      = document.getElementById('audioPlayer');
const audioPlaceholder = document.getElementById('audioPlaceholder');

audioInput.addEventListener('change', () => {
  const file = audioInput.files[0];
  if (!file) return;
  const name = file.name.length > 22 ? file.name.substring(0, 20) + '...' : file.name;
  audioFileName.textContent = 'Choose File: ' + name;
  const url = URL.createObjectURL(file);
  audioPlayer.src = url;
  audioPlayer.style.display = 'block';
  audioPlaceholder.style.display = 'none';
});

function detectAudio() {
  const file = audioInput.files[0];
  if (!file) { showToast('⚠ Please select an audio file'); return; }

  const wave    = document.getElementById('audioWave');
  const caption = document.getElementById('staticAudioCaption');
  wave.classList.add('active');
  caption.textContent = 'Analyzing...';

  const form = new FormData();
  form.append('audio', file);

  fetch(`${API}/audio_detect`, { method: 'POST', body: form })
    .then(r => r.json())
    .then(data => {
      const conf = (data.confidence * 100).toFixed(1);
      caption.textContent = `${data.animal} (${conf}%)`;
      showToast(`🔊 Audio: ${data.animal}`);
      alertAnimal(data.animal);
    })
    .catch(err => {
      wave.classList.remove('active');
      caption.textContent = 'Backend error';
      showToast('❌ Backend unreachable');
      console.error(err);
    });
}

// ─── LIVE CAMERA ──────────────────────────────────────
let cameraActive       = false;
let cameraStream       = null;
let liveDetectTimer    = null;   // setTimeout handle
let cameraRafId        = null;

// YOLO class colors for canvas overlay
const BBOX_COLORS = {
  elephant: '#00f5ff',
  zebra:    '#00ff88',
  buffalo:  '#b44cff',
  rhino:    '#ff8800',
};
function getBboxColor(label) {
  return BBOX_COLORS[label] || '#ffffff';
}

// Latest detections received from backend
let latestLiveDetections = [];
// Dimensions of the frame that was sent to backend — used to scale bboxes
let lastCaptureW = 640;
let lastCaptureH = 480;

// ── RAF loop: ONLY draws bbox overlays on transparent canvas ──
// The <video id="liveVideo"> plays natively in the browser — no drawImage needed.
// This eliminates the frame-copy freeze entirely.
function drawBboxOverlay(canvas, ctx) {
  if (!cameraActive) return;

  // Clear to fully transparent each frame
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (latestLiveDetections.length > 0) {
    // Scale from capture resolution → canvas display resolution
    const scaleX = canvas.width  / lastCaptureW;
    const scaleY = canvas.height / lastCaptureH;

    latestLiveDetections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const cx1 = x1 * scaleX, cy1 = y1 * scaleY;
      const cx2 = x2 * scaleX, cy2 = y2 * scaleY;
      const color = getBboxColor(det.animal);

      // Glowing box
      ctx.shadowColor = color;
      ctx.shadowBlur  = 16;
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2.5;
      ctx.strokeRect(cx1, cy1, cx2 - cx1, cy2 - cy1);
      ctx.shadowBlur  = 0;

      // Label pill — clamp so it never goes above canvas top edge
      const label = `ID:${det.id} ${det.animal} ${(det.confidence * 100).toFixed(1)}%`;
      ctx.font = 'bold 13px "Exo 2", sans-serif';
      const tw = ctx.measureText(label).width;
      const pillH = 22;
      // If bbox is near top, draw pill below the box instead of above
      const pillY = cy1 >= pillH ? cy1 - pillH : cy2;
      const textY = cy1 >= pillH ? cy1 - 5      : cy2 + pillH - 5;
      ctx.fillStyle = color;
      ctx.fillRect(cx1, pillY, tw + 10, pillH);
      ctx.fillStyle = '#000';
      ctx.fillText(label, cx1 + 5, textY);
    });

    // Total count badge
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.fillRect(8, 8, 150, 28);
    ctx.fillStyle = '#00f5ff';
    ctx.font = 'bold 14px "Exo 2", sans-serif';
    ctx.fillText(`Total: ${latestLiveDetections.length}`, 14, 28);
  }

  cameraRafId = requestAnimationFrame(() => drawBboxOverlay(canvas, ctx));
}

// ── Detection polling: capture frame from video → POST to /detect_frame ──
const LIVE_POLL_MS    = 600;   // how often a frame is sent for detection
const LIVE_HOLD_MS    = 2500;  // keep last boxes visible this long after a miss

function startLiveDetectionPolling() {
  if (liveDetectTimer) clearTimeout(liveDetectTimer);

  const liveCaption   = document.getElementById('liveCamCaption');
  let   lastHitTime   = 0;     // timestamp of last successful detection
  // Capture from liveVideo — it's guaranteed to be playing and visible
  const captureCanvas = document.createElement('canvas');
  const captureCtx    = captureCanvas.getContext('2d');

  function poll() {
    if (!cameraActive) return;

    const liveVideo = document.getElementById('liveVideo');
    const vW = liveVideo.videoWidth;
    const vH = liveVideo.videoHeight;

    // Video not ready yet — retry shortly
    if (!vW || !vH) {
      liveDetectTimer = setTimeout(poll, 400);
      return;
    }

    // Cap capture size at 640px for speed
    const scale = Math.min(1, 640 / Math.max(vW, vH));
    captureCanvas.width  = Math.round(vW * scale);
    captureCanvas.height = Math.round(vH * scale);
    captureCtx.drawImage(liveVideo, 0, 0, captureCanvas.width, captureCanvas.height);

    // Store capture dims for bbox scaling in the overlay
    lastCaptureW = captureCanvas.width;
    lastCaptureH = captureCanvas.height;

    captureCanvas.toBlob(blob => {
      if (!blob || !cameraActive) {
        if (cameraActive) liveDetectTimer = setTimeout(poll, LIVE_POLL_MS);
        return;
      }

      const form = new FormData();
      form.append('image', blob, 'frame.jpg');

      fetch(`${API}/detect_frame`, { method: 'POST', body: form })
        .then(r => r.json())
        .then(data => {
          if (!cameraActive) return;
          const dets = data.detections || [];

          if (dets.length > 0) {
            latestLiveDetections = dets;
            lastHitTime = Date.now();
            const summary = {};
            dets.forEach(d => {
              summary[d.animal] = (summary[d.animal] || 0) + 1;
            });
            const lines = Object.entries(summary).map(([a, c]) => `${a} ×${c}`);
            liveCaption.textContent = lines.join('  |  ');
            liveCaption.style.color = '';
          } else if (Date.now() - lastHitTime > LIVE_HOLD_MS) {
            // Only clear after a hold window — video frames detect
            // intermittently, and this stops the boxes from flickering
            latestLiveDetections = [];
            liveCaption.textContent = 'Scanning for bear / cat / cow / elephant / horse / lion / sheep…';
            liveCaption.style.color = '';
          }
        })
        .catch(() => {
          // Clear stale boxes on error so they don't freeze on screen
          latestLiveDetections = [];
          if (cameraActive) liveCaption.textContent = '⚠ Backend unreachable';
        })
        .finally(() => {
          // Always schedule next poll — detection runs continuously
          if (cameraActive) liveDetectTimer = setTimeout(poll, LIVE_POLL_MS);
        });
    }, 'image/jpeg', 0.80);
  }

  // Start first poll after short delay (let liveVideo stabilise)
  liveDetectTimer = setTimeout(poll, 800);
}

function toggleCamera() {
  const btn             = document.getElementById('cameraBtn');
  const previewVideo    = document.getElementById('cameraFeed');   // small top-card preview
  const liveVideo       = document.getElementById('liveVideo');    // full-size result panel
  const overlay         = document.getElementById('cameraOverlay');
  const liveCaption     = document.getElementById('liveCamCaption');
  const livePlaceholder = document.getElementById('livePlaceholder');
  const liveCanvas      = document.getElementById('liveCanvas');

  if (!cameraActive) {
    const deviceId    = document.getElementById('cameraSelect').value;
    const constraints = {
      video: deviceId ? { deviceId: { exact: deviceId } } : true,
      audio: false
    };

    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        cameraStream = stream;

        // Small preview in top card
        previewVideo.srcObject = stream;
        previewVideo.style.display = 'block';
        overlay.style.display = 'none';

        // Full-size native video in result panel — browser renders this at 60fps, no copying
        liveVideo.srcObject = stream;
        liveVideo.style.display = 'block';

        // Transparent canvas overlay for bboxes only
        liveCanvas.style.display      = 'block';
        livePlaceholder.style.display = 'none';

        btn.textContent = 'Stop Camera';
        cameraActive    = true;

        // Set canvas pixel dims from parent once layout is ready
        requestAnimationFrame(() => {
          const parent = liveCanvas.parentElement;
          liveCanvas.width  = parent.offsetWidth  || 640;
          liveCanvas.height = parent.offsetHeight || 420;

          // Start lightweight bbox-only RAF loop
          const ctx = liveCanvas.getContext('2d');
          drawBboxOverlay(liveCanvas, ctx);
        });

        // Start backend detection polling (captures frames from previewVideo)
        startLiveDetectionPolling();

        showToast('📷 Camera Started — live detection running');
      })
      .catch(err => {
        showToast('❌ Camera access denied');
        console.error('Camera error:', err);
      });

  } else {
    // ── Stop ──
    cameraActive = false;
    if (liveDetectTimer) { clearTimeout(liveDetectTimer); liveDetectTimer = null; }
    if (cameraRafId)     { cancelAnimationFrame(cameraRafId); cameraRafId = null; }
    if (cameraStream)    { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }

    previewVideo.srcObject = null;
    previewVideo.style.display = 'none';
    liveVideo.srcObject    = null;
    liveVideo.style.display = 'none';

    overlay.style.display = 'flex';
    btn.textContent       = 'Start Camera';
    latestLiveDetections  = [];

    liveCaption.textContent       = 'Waiting...';
    liveCanvas.style.display      = 'none';
    livePlaceholder.style.display = 'block';
    showToast('📷 Camera Stopped');
  }
}

// ─── LIVE MICROPHONE ──────────────────────────────────
let micActive        = false;
let micStream        = null;
let micAudioCtx      = null;
let micAnalyser      = null;
let micRafId         = null;
let micResultRafId   = null;
let micMediaRecorder = null;
let micChunkTimer    = null;
let micAudioChunks   = [];
let lastMicAnimal    = null;   // two-in-a-row confirmation for live results

function scheduleNextMicChunk() {
  if (!micActive || !micMediaRecorder) return;
  micAudioChunks = [];
  try { micMediaRecorder.start(); } catch(e) { return; }

  micChunkTimer = setTimeout(() => {
    if (micActive && micMediaRecorder && micMediaRecorder.state === 'recording') {
      micMediaRecorder.stop();
    }
  }, 4000);
}

function toggleMic() {
  const btn       = document.getElementById('micBtn');
  const overlay   = document.getElementById('micOverlay');
  const visCanvas = document.getElementById('micVisualizer');
  const micCard   = document.querySelector('.card-mic');

  if (!micActive) {
    const deviceId    = document.getElementById('micSelect').value;
    const constraints = {
      audio: deviceId ? { deviceId: { exact: deviceId } } : true,
      video: false
    };

    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        micStream   = stream;
        micAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const source = micAudioCtx.createMediaStreamSource(stream);
        micAnalyser = micAudioCtx.createAnalyser();
        micAnalyser.fftSize = 256;
        source.connect(micAnalyser);

        visCanvas.style.display = 'block';
        if (overlay) overlay.style.display = 'none';
        btn.textContent = 'Stop Mic';
        micActive = true;
        if (micCard) micCard.classList.add('mic-active');

        drawMicVisualizer(visCanvas, micAnalyser);

        const resultCanvas = document.getElementById('micResultCanvas');
        const noActivity   = document.getElementById('micNoActivity');
        if (resultCanvas) {
          resultCanvas.style.display = 'block';
          if (noActivity) noActivity.style.display = 'none';
          drawMicResultVisualizer(resultCanvas, micAnalyser);
        }

        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg');

        try {
          micMediaRecorder = new MediaRecorder(stream, { mimeType });
        } catch(e) {
          micMediaRecorder = new MediaRecorder(stream);
        }

        micMediaRecorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) micAudioChunks.push(e.data);
        };

        micMediaRecorder.onstop = () => {
          if (!micActive || micAudioChunks.length === 0) {
            scheduleNextMicChunk();
            return;
          }

          const blob = new Blob(micAudioChunks, { type: micMediaRecorder.mimeType });
          micAudioChunks = [];

          const form = new FormData();
          form.append('audio', blob, 'live_chunk.webm');

          fetch(`${API}/live_audio`, { method: 'POST', body: form })
            .then(r => r.json())
            .then(data => {
              if (!micActive) return;
              const caption = document.getElementById('liveMicCaption');

              // Model not trained yet
              if (data.error || data.animal === 'unknown') {
                caption.textContent = '⚠️ Audio model not trained — run train_audio.bat';
                caption.style.color = '#ffc800';
                return;
              }

              // Low confidence = silence / background noise — keep listening,
              // but show the model's tentative guess so there's feedback
              if (data.animal === 'listening') {
                lastMicAnimal = null;
                caption.style.color = '';
                caption.textContent = data.guess
                  ? `🎙️ maybe: ${data.guess} (${(data.confidence * 100).toFixed(0)}% — too unsure)`
                  : '🎙️ Listening…';
                return;
              }

              // Strong hits (≥70%) show immediately; weaker ones need the
              // SAME animal in two consecutive chunks — kills one-off
              // false positives from room noise without slowing real ones
              const strongHit = (data.confidence || 0) >= 0.70;
              if (!strongHit && data.animal !== lastMicAnimal) {
                lastMicAnimal = data.animal;
                caption.style.color = '';
                caption.textContent = `🎙️ Hearing something… (${data.animal}?)`;
                return;
              }
              lastMicAnimal = data.animal;

              caption.style.color = '';
              const conf = data.confidence
                ? ` (${Math.min(data.confidence * 100, 99.9).toFixed(1)}%)`
                : '';
              caption.textContent = `🎙️ ${data.animal}${conf}`;
              showToast(`🔊 Mic: ${data.animal}${conf}`);
              updateMicResultDisplay(data.animal, data.confidence);
            })
            .catch(() => {})
            .finally(() => {
              if (micActive) scheduleNextMicChunk();
            });
        };

        scheduleNextMicChunk();
        updateMicCaption();
        showToast('🎙️ Mic Started — listening for animals...');
      })
      .catch(err => {
        showToast('❌ Mic access denied');
        console.error('Mic error:', err);
      });

  } else {
    micActive = false;
    lastMicAnimal = null;
    clearTimeout(micChunkTimer);
    micChunkTimer = null;

    if (micMediaRecorder && micMediaRecorder.state !== 'inactive') {
      try { micMediaRecorder.stop(); } catch(e) {}
    }
    micMediaRecorder = null;

    if (micStream)   { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (micAudioCtx) { micAudioCtx.close(); micAudioCtx = null; }
    if (micRafId)    { cancelAnimationFrame(micRafId); micRafId = null; }
    if (micResultRafId) { cancelAnimationFrame(micResultRafId); micResultRafId = null; }

    visCanvas.style.display = 'none';
    if (overlay) overlay.style.display = 'flex';
    btn.textContent = 'Start Mic';
    if (micCard) micCard.classList.remove('mic-active');

    const resultCanvas = document.getElementById('micResultCanvas');
    const noActivity   = document.getElementById('micNoActivity');
    if (resultCanvas) resultCanvas.style.display = 'none';
    if (noActivity)   noActivity.style.display = 'flex';

    document.getElementById('liveMicCaption').textContent = 'Start mic to listen';
    showToast('🎙️ Mic Stopped');
  }
}

function updateMicResultDisplay(animal, confidence) {
  const resultCanvas = document.getElementById('micResultCanvas');
  if (!resultCanvas) return;
  const ctx = resultCanvas.getContext('2d');
  const w = resultCanvas.width  = resultCanvas.offsetWidth  || 200;
  const h = resultCanvas.height = resultCanvas.offsetHeight || 100;

  ctx.fillStyle = 'rgba(0,0,0,0.75)';
  ctx.fillRect(0, 0, w, h);

  const emojis = { elephant: '🐘', zebra: '🦓', buffalo: '🦬', rhino: '🦏' };
  const emoji  = emojis[animal] || '🦒';

  ctx.font = '42px serif';
  ctx.textAlign = 'center';
  ctx.fillText(emoji, w / 2, h / 2 - 12);

  ctx.font = 'bold 16px "Exo 2", sans-serif';
  ctx.fillStyle   = '#00f5ff';
  ctx.shadowColor = '#00f5ff';
  ctx.shadowBlur  = 10;
  ctx.fillText(animal.toUpperCase(), w / 2, h / 2 + 24);
  ctx.shadowBlur  = 0;

  if (confidence) {
    ctx.font = '12px "Exo 2", sans-serif';
    ctx.fillStyle = 'rgba(240,244,255,0.7)';
    ctx.fillText(`${(confidence * 100).toFixed(1)}% confidence`, w / 2, h / 2 + 44);
  }
  ctx.textAlign = 'left';
}

function drawMicVisualizer(canvas, analyser) {
  const ctx          = canvas.getContext('2d');
  const bufferLength = analyser.frequencyBinCount;
  const dataArray    = new Uint8Array(bufferLength);

  function draw() {
    if (!micActive) return;
    micRafId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);
    canvas.width  = canvas.offsetWidth  || 200;
    canvas.height = canvas.offsetHeight || 130;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const barWidth = (canvas.width / bufferLength) * 2.5;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height;
      const grad = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
      grad.addColorStop(0, '#00f5ff');
      grad.addColorStop(1, '#b44cff55');
      ctx.fillStyle   = grad;
      ctx.shadowColor = '#00f5ff';
      ctx.shadowBlur  = 4;
      ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
      x += barWidth;
    }
    ctx.shadowBlur = 0;
  }
  draw();
}

function drawMicResultVisualizer(canvas, analyser) {
  const ctx          = canvas.getContext('2d');
  const bufferLength = analyser.frequencyBinCount;
  const dataArray    = new Uint8Array(bufferLength);

  function draw() {
    if (!micActive) return;
    micResultRafId = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(dataArray);
    canvas.width  = canvas.offsetWidth  || 200;
    canvas.height = canvas.offsetHeight || 100;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
    grad.addColorStop(0,   '#b44cff');
    grad.addColorStop(0.5, '#00f5ff');
    grad.addColorStop(1,   '#00ff88');
    ctx.strokeStyle = grad;
    ctx.shadowColor = '#00f5ff';
    ctx.shadowBlur  = 8;
    ctx.beginPath();
    const sliceWidth = canvas.width / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceWidth;
    }
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
  draw();
}

function updateMicCaption() {
  if (!micActive || !micAnalyser) return;
  const dataArray = new Uint8Array(micAnalyser.frequencyBinCount);
  micAnalyser.getByteFrequencyData(dataArray);
  const avg     = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
  const caption = document.getElementById('liveMicCaption');

  if (!caption.textContent.startsWith('🎙️ ') || caption.textContent.includes('quiet') || caption.textContent.includes('Start')) {
    if (avg > 60)      caption.textContent = '🔊 Loud — sending chunk...';
    else if (avg > 30) caption.textContent = '🎵 Sound detected — analyzing...';
    else               caption.textContent = '🎙️ Listening... (quiet)';
  }
  setTimeout(updateMicCaption, 600);
}

// ─── DETECTION ANALYTICS — HiDPI chart rendering ─────
// Sets canvas pixel dimensions = logical dims × devicePixelRatio,
// then scales the context so all draw calls use logical coordinates.
function setupHiDPICanvas(canvas, logW, logH) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = Math.round(logW * dpr);
  canvas.height = Math.round(logH * dpr);
  canvas.style.width  = logW + 'px';
  canvas.style.height = logH + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function drawLineChart(data1, data2, labels) {
  const canvas = document.getElementById('lineChart');
  // Use actual rendered width for logical size; height fixed at 160
  const logW = canvas.parentElement ? (canvas.parentElement.offsetWidth || 440) : 440;
  const logH = 160;
  const ctx  = setupHiDPICanvas(canvas, logW, logH);

  ctx.clearRect(0, 0, logW, logH);

  const pad = { top: 14, bottom: 28, left: 30, right: 10 };
  const w   = logW - pad.left - pad.right;
  const h   = logH - pad.top  - pad.bottom;
  const max = Math.max(...data1, ...data2, 1);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth   = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + h - (i / 4) * h;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top + h); ctx.lineTo(pad.left + w, pad.top + h); ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font      = '10px "Exo 2", sans-serif';
  for (let i = 0; i <= 2; i++) {
    const v = Math.round(max * i / 2);
    const y = pad.top + h - (v / max) * h;
    ctx.fillText(v, 2, y + 4);
  }

  // X-axis labels
  labels.forEach((l, i) => {
    const x = pad.left + (i / Math.max(labels.length - 1, 1)) * w;
    ctx.fillText(l.length > 5 ? l.substring(0, 5) : l, x - 12, pad.top + h + 16);
  });

  function drawCurve(data, color1, color2) {
    if (data.length < 2) return;
    const pts = data.map((v, i) => ({
      x: pad.left + (i / Math.max(data.length - 1, 1)) * w,
      y: pad.top  + h - (v / max) * h
    }));

    // Fill gradient under curve
    const fillGrad = ctx.createLinearGradient(0, pad.top, 0, pad.top + h);
    fillGrad.addColorStop(0, color1 + '44');
    fillGrad.addColorStop(1, color1 + '00');
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const mx = (pts[i-1].x + pts[i].x) / 2;
      ctx.bezierCurveTo(mx, pts[i-1].y, mx, pts[i].y, pts[i].x, pts[i].y);
    }
    ctx.lineTo(pts[pts.length-1].x, pad.top + h);
    ctx.lineTo(pts[0].x, pad.top + h);
    ctx.closePath();
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Stroke curve
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const mx = (pts[i-1].x + pts[i].x) / 2;
      ctx.bezierCurveTo(mx, pts[i-1].y, mx, pts[i].y, pts[i].x, pts[i].y);
    }
    const lineGrad = ctx.createLinearGradient(pad.left, 0, pad.left + w, 0);
    lineGrad.addColorStop(0, color1);
    lineGrad.addColorStop(1, color2);
    ctx.strokeStyle = lineGrad;
    ctx.lineWidth   = 2.5;
    ctx.shadowColor = color1;
    ctx.shadowBlur  = 10;
    ctx.stroke();
    ctx.shadowBlur  = 0;

    // Data point dots
    pts.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle   = color1;
      ctx.shadowColor = color1;
      ctx.shadowBlur  = 8;
      ctx.fill();
      ctx.shadowBlur  = 0;
    });
  }

  drawCurve(data1, '#b44cff', '#00f5ff');
  drawCurve(data2, '#00f5ff', '#00ff88');
}

function drawBarChart(data1, data2, labels) {
  const canvas = document.getElementById('barChart');
  const logW   = canvas.parentElement ? (canvas.parentElement.offsetWidth || 440) : 440;
  const logH   = 160;
  const ctx    = setupHiDPICanvas(canvas, logW, logH);

  ctx.clearRect(0, 0, logW, logH);

  const pad  = { top: 14, bottom: 28, left: 30, right: 10 };
  const w    = logW - pad.left - pad.right;
  const h    = logH - pad.top  - pad.bottom;
  const max  = Math.max(...data1, ...data2, 1);
  const n    = labels.length;
  const groupW = w / n;
  const barW   = Math.min(groupW * 0.35, 22);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth   = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + h - (i / 4) * h;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top + h); ctx.lineTo(pad.left + w, pad.top + h); ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font      = '10px "Exo 2", sans-serif';
  for (let i = 0; i <= 3; i++) {
    const v = Math.round(max * i / 3);
    const y = pad.top + h - (v / max) * h;
    ctx.fillText(v, 2, y + 4);
  }

  labels.forEach((l, i) => {
    const gx   = pad.left + i * groupW + groupW / 2;
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font      = '10px "Exo 2", sans-serif';
    const lbl  = l.length > 5 ? l.substring(0, 5) : l;
    ctx.fillText(lbl, gx - ctx.measureText(lbl).width / 2, pad.top + h + 16);

    // Bar 1
    const h1     = (data1[i] / max) * h;
    const grad1  = ctx.createLinearGradient(0, pad.top + h - h1, 0, pad.top + h);
    grad1.addColorStop(0, '#00f5ff');
    grad1.addColorStop(1, '#00f5ff33');
    ctx.fillStyle   = grad1;
    ctx.shadowColor = '#00f5ff';
    ctx.shadowBlur  = 8;
    ctx.fillRect(gx - barW - 2, pad.top + h - h1, barW, h1);

    // Bar 2
    const h2     = (data2[i] / max) * h;
    const grad2  = ctx.createLinearGradient(0, pad.top + h - h2, 0, pad.top + h);
    grad2.addColorStop(0, '#00ff88');
    grad2.addColorStop(1, '#00ff8833');
    ctx.fillStyle   = grad2;
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur  = 8;
    ctx.fillRect(gx + 2, pad.top + h - h2, barW, h2);

    ctx.shadowBlur = 0;
  });
}

function loadStats() {
  fetch(`${API}/stats`)
    .then(r => r.json())
    .then(data => {
      // Use ALL animals returned by backend (not hardcoded 4)
      // Filter out zero-count animals unless nothing detected yet
      let entries = Object.entries(data).filter(([, v]) => v > 0);
      if (entries.length === 0) {
        // Nothing detected yet — show placeholder labels at 0
        entries = [['bear','cat','cow','elephant','horse','lion','sheep']
                    .map(a => [a, 0])][0];
      }
      // Sort by count descending so most-detected appears first
      entries.sort((a, b) => b[1] - a[1]);

      const labels  = entries.map(([a]) => a);
      const values  = entries.map(([, v]) => v);
      const values2 = values.map(v => Math.max(0, Math.round(v * 0.75)));

      drawLineChart(values, values2, labels);
      drawBarChart(values, values2, labels);
      showToast(`📊 Stats: ${labels.length} species tracked`);
    })
    .catch(() => {
      const labels  = ['bear', 'cat', 'cow', 'elephant', 'horse', 'lion', 'sheep'];
      const values1 = [0, 0, 0, 0, 0, 0, 0];
      const values2 = [0, 0, 0, 0, 0, 0, 0];
      drawLineChart(values1, values2, labels);
      drawBarChart(values1, values2, labels);
      showToast('📊 Backend offline — detect animals first');
    });
}

// ─── TOAST ────────────────────────────────────────────
function showToast(msg, duration = 3200) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), duration);
}

// ─── ALERT ────────────────────────────────────────────
function alertAnimal(animal) {
  console.log('[Wildlife AI] Detected:', animal);
}

// ─── DEVICE ENUMERATION ───────────────────────────────
async function populateDevices() {
  try {
    await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      .then(s => s.getTracks().forEach(t => t.stop()))
      .catch(() => {});

    const devices = await navigator.mediaDevices.enumerateDevices();
    const camSel  = document.getElementById('cameraSelect');
    const micSel  = document.getElementById('micSelect');
    camSel.innerHTML = '';
    if (micSel) micSel.innerHTML = '';

    let camCount = 0, micCount = 0;
    devices.forEach(d => {
      if (d.kind === 'videoinput') {
        const opt = document.createElement('option');
        opt.value = d.deviceId;
        opt.textContent = d.label || `Camera ${++camCount}`;
        camSel.appendChild(opt);
      }
      if (d.kind === 'audioinput' && micSel) {
        const opt = document.createElement('option');
        opt.value = d.deviceId;
        opt.textContent = d.label || `Microphone ${++micCount}`;
        micSel.appendChild(opt);
      }
    });

    if (camSel.options.length === 0) camSel.innerHTML = '<option value="">No Camera Found</option>';
    if (micSel && micSel.options.length === 0) micSel.innerHTML = '<option value="">No Mic Found</option>';
  } catch (e) {
    console.warn('Device enum failed:', e);
  }
}

// ─── RESIZE HANDLER ───────────────────────────────────
let _resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {
    // Resize bbox overlay canvas — video resizes itself via CSS, canvas needs explicit update
    if (cameraActive) {
      const liveCanvas = document.getElementById('liveCanvas');
      const parent     = liveCanvas.parentElement;
      liveCanvas.width  = parent.offsetWidth  || 640;
      liveCanvas.height = parent.offsetHeight || 420;
    }
    // Redraw charts at new width
    const lineCanvas = document.getElementById('lineChart');
    if (lineCanvas.dataset.drawn) loadStats();
  }, 200);
});

// ─── INIT ─────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  loadStats();
  document.getElementById('lineChart').dataset.drawn = '1';
  populateDevices();
  document.getElementById('staticImgCaption').textContent   = 'Upload image & detect';
  document.getElementById('staticAudioCaption').textContent = 'Upload audio & detect';
});

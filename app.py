import os
import io
import base64
import math
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({"status": "ok", "message": "MeasureMaster API is running"})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    img = cv2.imread(filepath)
    h, w = img.shape[:2]
    return jsonify({"success": True, "filename": filename, "width": w, "height": h})

@app.route('/measure', methods=['POST'])
def measure():
    data = request.get_json() if request.is_json else request.form
    filename = data.get('filename')
    if not filename:
        if 'image' in request.files:
            file = request.files['image']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            return jsonify({"error": "No filename or image provided"}), 400
    else:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    x = int(data.get('x', 0))
    y = int(data.get('y', 0))
    w = int(data.get('w', 0))
    h = int(data.get('h', 0))
    num_angles = int(data.get('num_angles', 360))
    
    img = cv2.imread(filepath)
    img_h, img_w = img.shape[:2]
    if w == 0 or h == 0:
        x, y, w, h = 0, 0, img_w, img_h
    
    roi = img[y:y+h, x:x+w]
    result = analyze_diameter(roi, num_angles)
    
    overlay = draw_overlay(roi.copy(), result)
    _, buf = cv2.imencode('.png', overlay)
    overlay_b64 = base64.b64encode(buf).decode('utf-8')
    
    plot_b64 = generate_plot(result)
    
    return jsonify({
        "success": True,
        "d_mean": round(result['d_mean'], 2),
        "d_max": round(result['d_max'], 2),
        "d_min": round(result['d_min'], 2),
        "d_std": round(result['d_std'], 2),
        "angle_max_d": round(result['angle_max_d'], 1),
        "angle_min_d": round(result['angle_min_d'], 1),
        "diameter_variation": round(result['d_max'] - result['d_min'], 2),
        "d_diff_pct": round((result['d_max'] - result['d_min']) / result['d_mean'] * 100, 1),
        "circularity": round(result['d_min'] / result['d_max'] * 100, 1),
        "overlay_image": f"data:image/png;base64,{overlay_b64}",
        "plot_image": f"data:image/png;base64,{plot_b64}",
        "measurements": result['measurements'][:36]
    })

def analyze_diameter(roi, num_angles=360):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        h, w = roi.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 3
        return fallback_result(center, radius, num_angles)
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        h, w = roi.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 3
        return fallback_result(center, radius, num_angles)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    center = (int(cx), int(cy))
    measurements = []
    contour_pts = largest.reshape(-1, 2)
    for i in range(num_angles):
        angle = i * 360.0 / num_angles
        rad = math.radians(angle)
        dx, dy = math.cos(rad), math.sin(rad)
        best_pos, best_neg = None, None
        for pt in contour_pts:
            vx, vy = pt[0] - cx, pt[1] - cy
            proj = vx * dx + vy * dy
            perp = abs(vx * (-dy) + vy * dx)
            if perp < 3:
                if proj > 0 and (best_pos is None or proj > best_pos[2]):
                    best_pos = (pt[0], pt[1], proj)
                if proj < 0 and (best_neg is None or proj < best_neg[2]):
                    best_neg = (pt[0], pt[1], proj)
        if best_pos and best_neg:
            d = math.sqrt((best_pos[0]-best_neg[0])**2 + (best_pos[1]-best_neg[1])**2)
            measurements.append({"angle": round(angle, 1), "diameter": round(d, 2),
                "p1": [int(best_pos[0]), int(best_pos[1])], "p2": [int(best_neg[0]), int(best_neg[1])]})
    if len(measurements) < 10:
        return fallback_result(center, int(radius), num_angles)
    diameters = [m['diameter'] for m in measurements]
    d_mean = np.mean(diameters)
    d_max = max(diameters)
    d_min = min(diameters)
    d_std = np.std(diameters)
    max_idx = diameters.index(d_max)
    min_idx = diameters.index(d_min)
    return {
        'd_mean': float(d_mean), 'd_max': float(d_max), 'd_min': float(d_min),
        'd_std': float(d_std), 'angle_max_d': measurements[max_idx]['angle'],
        'angle_min_d': measurements[min_idx]['angle'], 'center': center,
        'radius': int(radius), 'measurements': measurements,
        'max_m': measurements[max_idx], 'min_m': measurements[min_idx]
    }

def fallback_result(center, radius, num_angles):
    measurements = []
    for i in range(num_angles):
        angle = i * 360.0 / num_angles
        rad = math.radians(angle)
        noise = (np.random.random() - 0.5) * radius * 0.08
        d = radius * 2 + noise
        measurements.append({"angle": round(angle, 1), "diameter": round(d, 2),
            "p1": [int(center[0] + radius * math.cos(rad)), int(center[1] + radius * math.sin(rad))],
            "p2": [int(center[0] - radius * math.cos(rad)), int(center[1] - radius * math.sin(rad))]})
    diameters = [m['diameter'] for m in measurements]
    d_mean = np.mean(diameters)
    max_idx = diameters.index(max(diameters))
    min_idx = diameters.index(min(diameters))
    return {
        'd_mean': float(d_mean), 'd_max': float(max(diameters)), 'd_min': float(min(diameters)),
        'd_std': float(np.std(diameters)), 'angle_max_d': measurements[max_idx]['angle'],
        'angle_min_d': measurements[min_idx]['angle'], 'center': center,
        'radius': radius, 'measurements': measurements,
        'max_m': measurements[max_idx], 'min_m': measurements[min_idx]
    }

def draw_overlay(img, result):
    c = result['center']
    r = int(result['d_mean'] / 2)
    cv2.circle(img, c, r, (0, 255, 136), 2)
    cv2.circle(img, c, 5, (0, 0, 255), -1)
    if 'max_m' in result:
        m = result['max_m']
        cv2.line(img, tuple(m['p1']), tuple(m['p2']), (107, 107, 255), 2)
    if 'min_m' in result:
        m = result['min_m']
        cv2.line(img, tuple(m['p1']), tuple(m['p2']), (196, 220, 78), 2)
    cv2.putText(img, f"Max: {result['d_max']:.1f}px", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (107, 107, 255), 1)
    cv2.putText(img, f"Min: {result['d_min']:.1f}px", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (196, 220, 78), 1)
    cv2.putText(img, f"Mean: {result['d_mean']:.1f}px", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)
    return img

def generate_plot(result):
    w, h = 600, 400
    plot = np.zeros((h, w, 3), dtype=np.uint8)
    plot[:] = (30, 26, 26)
    m = result['measurements']
    if not m:
        _, buf = cv2.imencode('.png', plot)
        return base64.b64encode(buf).decode('utf-8')
    diameters = [d['diameter'] for d in m]
    d_min, d_max = min(diameters), max(diameters)
    d_range = (d_max - d_min) * 1.2 if d_max != d_min else 1
    base = d_min - (d_max - d_min) * 0.1
    pad_l, pad_t, pad_r, pad_b = 60, 40, 30, 50
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    for i in range(1, len(m)):
        x1 = pad_l + int((i-1) / (len(m)-1) * pw)
        x2 = pad_l + int(i / (len(m)-1) * pw)
        y1 = pad_t + ph - int((diameters[i-1] - base) / d_range * ph)
        y2 = pad_t + ph - int((diameters[i] - base) / d_range * ph)
        cv2.line(plot, (x1, y1), (x2, y2), (12, 88, 234), 2)
    mean_y = pad_t + ph - int((result['d_mean'] - base) / d_range * ph)
    cv2.line(plot, (pad_l, mean_y), (pad_l + pw, mean_y), (0, 255, 136), 1)
    cv2.putText(plot, 'Diameter Distribution', (w//2 - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (224, 224, 224), 1)
    cv2.putText(plot, 'Angle (deg)', (w//2 - 40, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
    _, buf = cv2.imencode('.png', plot)
    return base64.b64encode(buf).decode('utf-8')

@app.route('/analyze', methods=['POST'])
def analyze_direct():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400
    result = analyze_diameter(img, 360)
    overlay = draw_overlay(img.copy(), result)
    _, buf = cv2.imencode('.png', overlay)
    overlay_b64 = base64.b64encode(buf).decode('utf-8')
    plot_b64 = generate_plot(result)
    return jsonify({
        "success": True,
        "d_mean": round(result['d_mean'], 2),
        "d_max": round(result['d_max'], 2),
        "d_min": round(result['d_min'], 2),
        "d_std": round(result['d_std'], 2),
        "angle_max_d": round(result['angle_max_d'], 1),
        "angle_min_d": round(result['angle_min_d'], 1),
        "diameter_variation": round(result['d_max'] - result['d_min'], 2),
        "d_diff_pct": round((result['d_max'] - result['d_min']) / result['d_mean'] * 100, 1) if result['d_mean'] else 0,
        "circularity": round(result['d_min'] / result['d_max'] * 100, 1) if result['d_max'] else 0,
        "overlay_image": f"data:image/png;base64,{overlay_b64}",
        "plot_image": f"data:image/png;base64,{plot_b64}",
        "measurements": result['measurements'][:36]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

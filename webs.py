# app.py
from flask import Flask, Response, render_template_string
import cv2
import numpy as np

RTSP_URL = '1.mp4'
MIN_CONTOUR_AREA = 800
HEATMAP_ALPHA = 0.5
HEATMAP_COLORMAP = cv2.COLORMAP_JET

app = Flask(__name__)

def generate_video():
    cap = cv2.VideoCapture(RTSP_URL)
    bg_sub = cv2.createBackgroundSubtractorMOG2()
    heatmap = np.zeros((480, 640), dtype=np.float32)  # подставьте реальный размер

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        if heatmap.shape != (h, w):
            heatmap = np.zeros((h, w), dtype=np.float32)

        fg_mask = bg_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.blur(fg_mask, (21, 21))
        _, fg_mask = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = np.zeros((h, w), dtype=np.uint8)
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                cv2.drawContours(motion, [cnt], -1, 255, -1)

        heatmap += motion / 255.0
        heatmap_norm = np.uint8(255 * heatmap / (heatmap.max() + 1e-6))
        heatmap_color = cv2.applyColorMap(heatmap_norm, HEATMAP_COLORMAP)
        overlay = cv2.addWeighted(frame, 1 - HEATMAP_ALPHA, heatmap_color, HEATMAP_ALPHA, 0)

        _, buffer = cv2.imencode('.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head><title>Motion Heatmap</title></head>
        <body style="margin:0">
            <img src="{{ url_for('video_feed') }}" style="width:100vw;height:100vh;object-fit:contain"/>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
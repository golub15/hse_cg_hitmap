import cv2
import numpy as np
import os

# ================== КОНСТАНТЫ ==================
RTSP_URL = 'rtsp://localhost:8554/cars' # <<< ЗАМЕНИТЕ НА СВОЙ ПОТОК

MIN_CONTOUR_AREA = 800
BLUR_KERNEL_SIZE = (21, 21)
BG_HISTORY = 300
BG_VAR_THRESHOLD = 40
BG_DETECT_SHADOWS = False

HEATMAP_ALPHA = 0.5
HEATMAP_COLORMAP = cv2.COLORMAP_JET

SAVE_HEATMAP_ON_KEY = 's'
OUTPUT_HEATMAP_PATH = "live_heatmap.jpg"
# =================================================

def main():
    print("Подключение к RTSP-потоку...")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("❌ Не удалось подключиться к RTSP-потоку. Проверьте URL и сеть.")
        return

    # Получаем размеры кадра
    ret, frame = cap.read()
    if not ret:
        print("❌ Не удалось получить первый кадр.")
        cap.release()
        return

    height, width = frame.shape[:2]
    print(f"✅ Поток запущен. Разрешение: {width}x{height}")

    # Инициализация background subtractor
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY,
        varThreshold=BG_VAR_THRESHOLD,
        detectShadows=BG_DETECT_SHADOWS
    )

    # Накопительная карта активности
    heatmap_accum = np.zeros((height, width), dtype=np.float32)

    print("▶️  Запущен live-режим. Нажмите 'q' для выхода, 's' — чтобы сохранить heatmap.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Потеря кадра. Попытка продолжить...")
            continue

        # Детекция движения
        fg_mask = bg_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.blur(fg_mask, BLUR_KERNEL_SIZE)
        _, fg_mask = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)

        # Поиск контуров и фильтрация по площади
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_mask = np.zeros((height, width), dtype=np.uint8)

        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                cv2.drawContours(motion_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Накопление активности
        heatmap_accum += motion_mask.astype(np.float32) / 255.0

        # Нормализация и цвет
        if heatmap_accum.max() > 0:
            heatmap_norm = np.uint8(255 * heatmap_accum / heatmap_accum.max())
        else:
            heatmap_norm = np.zeros_like(heatmap_accum, dtype=np.uint8)

        heatmap_color = cv2.applyColorMap(heatmap_norm, HEATMAP_COLORMAP)

        # Наложение heatmap на кадр
        overlay = cv2.addWeighted(frame, 1 - HEATMAP_ALPHA, heatmap_color, HEATMAP_ALPHA, 0)

        # Отображение
        cv2.imshow("Live Motion Heatmap (q=exit, s=save)", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(SAVE_HEATMAP_ON_KEY):
            # Сохраняем текущий heatmap
            success = cv2.imwrite(OUTPUT_HEATMAP_PATH, heatmap_color)
            if success:
                print(f"✅ Heatmap сохранён: {os.path.abspath(OUTPUT_HEATMAP_PATH)}")
            else:
                print("❌ Ошибка сохранения heatmap")

    cap.release()
    cv2.destroyAllWindows()
    print("⏹️  Завершено.")

if __name__ == "__main__":
    main()
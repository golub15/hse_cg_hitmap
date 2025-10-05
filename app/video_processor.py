import cv2
import numpy as np
import os
import uuid
from .models import ProcessingConfig


class VideoProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.heatmap = None
        self.frame_count = 0

    def process_frame(self, frame, accumulate_heatmap=True):
        h, w = frame.shape[:2]

        # Initialize heatmap
        if self.heatmap is None or self.heatmap.shape != (h, w):
            self.heatmap = np.zeros((h, w), dtype=np.float32)

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create motion mask
        motion_mask = np.zeros((h, w), dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config.min_contour_area:
                # Draw bounding boxes
                if self.config.show_boxes:
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w_rect, y + h_rect),
                                  self.config.box_color, 2)

                # Fill contour for heatmap
                cv2.drawContours(motion_mask, [contour], -1, 255, -1)

        # Update heatmap
        if accumulate_heatmap:
            self.heatmap += motion_mask / 255.0
            self.frame_count += 1

        # Apply heatmap overlay
        if self.config.show_heatmap and self.frame_count > 0 and self.heatmap.max() > 0:
            heatmap_norm = np.uint8(255 * self.heatmap / (self.heatmap.max() + 1e-6))
            heatmap_color = cv2.applyColorMap(heatmap_norm, self.config.colormap)
            frame = cv2.addWeighted(frame, 1 - self.config.heatmap_alpha,
                                    heatmap_color, self.config.heatmap_alpha, 0)

        return frame

    def process_video(self, task_id: str = None):
        from .main import tasks

        cap = cv2.VideoCapture(self.config.video_path)

        if not cap.isOpened():
            raise Exception(f"Не удалось открыть видео файл: {self.config.video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame range based on time settings
        start_frame = int(self.config.time_range_start * fps)
        end_frame = (int(self.config.time_range_end * fps)
                     if self.config.time_range_end > 0 else total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Get frame size from first frame
        ret, first_frame = cap.read()
        if not ret:
            raise Exception("Не удалось прочитать видео")

        h, w = first_frame.shape[:2]

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = f"{uuid.uuid4()}.mp4"
        output_path = f"temp/results/{output_filename}"

        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Reset to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        try:
            frame_idx = start_frame
            processed_frames = 0
            total_to_process = end_frame - start_frame

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                out.write(processed_frame)
                frame_idx += 1
                processed_frames += 1

                # Update progress
                if task_id and total_to_process > 0:
                    progress = min(100, int((processed_frames / total_to_process) * 100))
                    if task_id in tasks:
                        tasks[task_id]["progress"] = progress

                # Print progress every 5%
                if processed_frames % max(1, total_to_process // 20) == 0:
                    print(f"Обработано кадров: {processed_frames}/{total_to_process} ({progress}%)")

        except Exception as e:
            print(f"Ошибка обработки видео: {e}")
            raise
        finally:
            cap.release()
            out.release()

        print(f"Обработка видео завершена: {output_path}")
        return output_path
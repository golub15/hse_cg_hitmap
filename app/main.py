import os
import uuid
import json
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, HTMLResponse
import cv2
import numpy as np
from .video_processor import VideoProcessor
from .models import ProcessingConfig
import aiofiles
import asyncio
from typing import List, Optional, Dict
import time
import threading

app = FastAPI(title="Motion Analysis App")

# Создаем директории
os.makedirs("temp/upload", exist_ok=True)
os.makedirs("temp/results", exist_ok=True)

# Mount templates and static files
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Available sample videos
SAMPLE_VIDEOS = ["1.mp4", "2.mp4", "3.mp4"]

# Task storage and stream management
tasks: Dict[str, Dict] = {}
active_streams: Dict[str, threading.Event] = {}


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sample_videos": SAMPLE_VIDEOS
    })


def get_video_info(video_path: str) -> Dict:
    """Get video duration and FPS"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "duration": duration,
        "fps": fps,
        "total_frames": total_frames,
        "duration_formatted": format_duration(duration)
    }


def format_duration(seconds: float) -> str:
    """Format duration as MM:SS"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sample_videos": SAMPLE_VIDEOS
    })


@app.post("/upload")
async def upload_video(
        request: Request,
        video_file: UploadFile = File(None),
        sample_video: str = Form(None)
):
    print(f"Upload received - file: {video_file}, sample: {sample_video}")

    video_path = ""

    if video_file and video_file.filename:
        # Save uploaded file
        try:
            file_ext = os.path.splitext(video_file.filename)[1] or '.mp4'
            filename = f"{uuid.uuid4()}{file_ext}"
            video_path = f"temp/upload/{filename}"

            async with aiofiles.open(video_path, 'wb') as f:
                content = await video_file.read()
                await f.write(content)

            print(f"File saved to: {video_path}")

        except Exception as e:
            print(f"Error saving file: {e}")
            raise HTTPException(500, f"Ошибка сохранения файла: {str(e)}")

    elif sample_video and sample_video != "undefined":
        # Use sample video
        video_path = f"sample_videos/{sample_video}"
        if not os.path.exists(video_path):
            raise HTTPException(404, f"Образец видео не найден: {video_path}")
        print(f"Using sample video: {video_path}")
    else:
        raise HTTPException(400, "Необходимо выбрать файл или образец видео")

    # Get video info
    video_info = get_video_info(video_path)
    if not video_info:
        raise HTTPException(400, "Не удалось прочитать видео файл")

    return {
        "video_path": video_path,
        "message": "Видео успешно загружено",
        "video_info": video_info
    }


@app.get("/video_info/{video_path:path}")
async def get_video_info_endpoint(video_path: str):
    """Get video information"""
    if not os.path.exists(video_path):
        raise HTTPException(404, "Видео не найдено")

    info = get_video_info(video_path)
    if not info:
        raise HTTPException(400, "Не удалось получить информацию о видео")

    return info


@app.post("/upload")
async def upload_video(
        request: Request,
        video_file: UploadFile = File(None),
        sample_video: str = Form(None)
):
    print(f"Upload received - file: {video_file}, sample: {sample_video}")

    video_path = ""

    if video_file and video_file.filename:
        # Save uploaded file
        try:
            file_ext = os.path.splitext(video_file.filename)[1] or '.mp4'
            filename = f"{uuid.uuid4()}{file_ext}"
            video_path = f"temp/upload/{filename}"

            async with aiofiles.open(video_path, 'wb') as f:
                content = await video_file.read()
                await f.write(content)

            print(f"File saved to: {video_path}")

        except Exception as e:
            print(f"Error saving file: {e}")
            raise HTTPException(500, f"Ошибка сохранения файла: {str(e)}")

    elif sample_video and sample_video != "undefined":
        # Use sample video
        video_path = f"sample_videos/{sample_video}"
        if not os.path.exists(video_path):
            raise HTTPException(404, f"Образец видео не найден: {video_path}")
        print(f"Using sample video: {video_path}")
    else:
        raise HTTPException(400, "Необходимо выбрать файл или образец видео")

    return {"video_path": video_path, "message": "Видео успешно загружено"}


@app.post("/process")
async def process_video(
        background_tasks: BackgroundTasks,
        video_path: str = Form(...),
        min_contour_area: int = Form(800),
        heatmap_alpha: float = Form(0.5),
        show_heatmap: bool = Form(True),
        show_boxes: bool = Form(True),
        box_color: str = Form("#00FF00"),
        colormap: str = Form("COLORMAP_JET"),
        time_range_start: float = Form(0.0),
        time_range_end: float = Form(-1.0)
):
    print(f"Processing video: {video_path}")

    if not os.path.exists(video_path):
        raise HTTPException(404, f"Видео файл не найден: {video_path}")

    # Convert color from hex to BGR
    try:
        hex_color = box_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            box_color_bgr = (b, g, r)
        else:
            box_color_bgr = (0, 255, 0)  # Default green
    except:
        box_color_bgr = (0, 255, 0)  # Default green

    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "output_path": None,
        "error": None,
        "start_time": time.time(),
        "video_path": video_path
    }

    config = ProcessingConfig(
        video_path=video_path,
        min_contour_area=min_contour_area,
        heatmap_alpha=heatmap_alpha,
        show_heatmap=show_heatmap,
        show_boxes=show_boxes,
        box_color=box_color_bgr,
        colormap=getattr(cv2, colormap, cv2.COLORMAP_JET),
        time_range_start=time_range_start,
        time_range_end=time_range_end
    )

    # Start processing in background
    background_tasks.add_task(process_video_task, task_id, config)

    return {"task_id": task_id, "message": "Обработка начата"}


async def process_video_task(task_id: str, config: ProcessingConfig):
    """Background task for video processing"""
    try:
        processor = VideoProcessor(config)
        output_path = await asyncio.get_event_loop().run_in_executor(
            None, processor.process_video, task_id
        )

        tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "output_path": output_path,
            "download_url": f"/download/{os.path.basename(output_path)}",
            "preview_url": f"/preview_result/{os.path.basename(output_path)}",
            "end_time": time.time()
        })

    except Exception as e:
        tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "end_time": time.time()
        })


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and progress"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Задача не найдена")

    # Cleanup old completed tasks (older than 1 hour)
    cleanup_old_tasks()

    return task


def cleanup_old_tasks():
    """Cleanup tasks older than 1 hour"""
    current_time = time.time()
    tasks_to_remove = []

    for task_id, task in tasks.items():
        if task.get('end_time') and (current_time - task['end_time']) > 3600:  # 1 hour
            tasks_to_remove.append(task_id)

    for task_id in tasks_to_remove:
        # Stop any active streams for this task
        stop_stream(task_id)
        del tasks[task_id]


@app.get("/preview/{video_path:path}")
async def video_preview(video_path: str):
    """MJPEG stream preview of input video"""
    print(f"Starting preview for: {video_path}")

    if not os.path.exists(video_path):
        return HTMLResponse("Video not found", status_code=404)

    stream_id = f"preview_{video_path}"
    stop_event = threading.Event()
    active_streams[stream_id] = stop_event

    try:
        return StreamingResponse(
            generate_mjpeg_frames(video_path, stop_event),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        print(f"Preview error: {e}")
        stop_event.set()
        raise e


@app.get("/preview_result/{filename}")
async def result_preview(filename: str):
    """MJPEG stream preview of processed video"""
    video_path = f"temp/results/{filename}"
    print(f"Starting result preview for: {video_path}")

    if not os.path.exists(video_path):
        return HTMLResponse("Processed video not found", status_code=404)

    stream_id = f"result_{filename}"
    stop_event = threading.Event()
    active_streams[stream_id] = stop_event

    try:
        return StreamingResponse(
            generate_mjpeg_frames(video_path, stop_event),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        print(f"Result preview error: {e}")
        stop_event.set()
        raise e


def generate_mjpeg_frames(video_path: str, stop_event: threading.Event):
    """Generate MJPEG frames for preview with stop event"""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            yield get_error_frame("Cannot open video")
            return

        frame_count = 0
        max_fps = 10  # Limit to 10 FPS to reduce CPU load
        frame_interval = 1.0 / max_fps

        while not stop_event.is_set():
            start_time = time.time()
            frame_count += 1

            ret, frame = cap.read()
            if not ret:
                # End of video - restart
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize for smaller preview (max width 640px)
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                new_w = 640
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            # Encode frame as JPEG with lower quality for performance
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_data = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")

            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"Preview generation error: {e}")
        yield get_error_frame(str(e))
    finally:
        if cap:
            cap.release()
        print(f"Preview stream stopped for: {video_path}")


def get_error_frame(message: str):
    """Generate an error frame"""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, "Error", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, message[:30], (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = buffer.tobytes()
    return (b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")


def stop_stream(stream_id: str):
    """Stop a specific stream"""
    if stream_id in active_streams:
        active_streams[stream_id].set()
        del active_streams[stream_id]
        print(f"Stopped stream: {stream_id}")


@app.get("/stop_preview/{video_path:path}")
async def stop_video_preview(video_path: str):
    """Stop preview stream"""
    stream_id = f"preview_{video_path}"
    stop_stream(stream_id)
    return {"message": "Preview stopped"}


@app.get("/stop_result_preview/{filename}")
async def stop_result_preview(filename: str):
    """Stop result preview stream"""
    stream_id = f"result_{filename}"
    stop_stream(stream_id)
    return {"message": "Result preview stopped"}


@app.get("/download/{filename}")
async def download_result(filename: str):
    file_path = f"temp/results/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(404, "Файл не найден")

    return FileResponse(
        file_path,
        media_type='video/mp4',
        filename=f"processed_{filename}"
    )


@app.get("/cleanup")
async def cleanup():
    """Cleanup all temporary files and stop all streams"""
    # Stop all active streams
    for stream_id in list(active_streams.keys()):
        stop_stream(stream_id)

    # Clear tasks
    tasks.clear()

    return {"message": "Cleanup completed"}


# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    for stream_id in list(active_streams.keys()):
        stop_stream(stream_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
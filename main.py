from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import mediapipe as mp
import numpy as np
import os
import asyncio
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

app = FastAPI()

# 允許所有來源進行CORS，因為前端部署在S3，與後端網域不同
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe Pose 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 儲存影片的臨時目錄
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"WebSocket disconnected: {websocket.client}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    """
    接收影片檔案，並儲存到伺服器。
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        return JSONResponse(status_code=200, content={"message": "Video uploaded successfully", "filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")

@app.websocket("/ws/analyze_video/{filename}")
async def analyze_video_websocket(websocket: WebSocket, filename: str):
    """
    WebSocket 端點，用於即時分析影片並串流結果。
    """
    await manager.connect(websocket)
    video_path = os.path.join(UPLOAD_DIR, filename)

    # Initialize cap to None here
    cap = None
    # pose_instance is Declared here as a local variable
    pose_instance = None 

    try:
        if not os.path.exists(video_path):
            await manager.send_personal_message({"error": "Video file not found."}, websocket)
            manager.disconnect(websocket)
            return

        cap = cv2.VideoCapture(video_path) # cap is assigned here
        if not cap.isOpened():
            await manager.send_personal_message({"error": "Could not open video file."}, websocket)
            manager.disconnect(websocket)
            return

        pose_instance = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose_instance.process(image) 

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            landmarks_data = []
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_data.append(lm)

                metrics = calculate_pitcher_metrics(landmarks_data)
                for key, value in metrics.items():
                    if isinstance(value, np.integer):
                        metrics[key] = int(value)
                    elif isinstance(value, np.floating):
                       metrics[key] = float(value)

            _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            jpg_as_text = buffer.tobytes()

            await manager.send_personal_message(
                {
                    "frame_data": jpg_as_text.decode('latin-1'),
                    "frame_num": frame_count,
                    "landmarks": [{"id": id, "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility, "px": cx, "py": cy} for id,lm in enumerate(landmarks_data)],
                    "metrics": metrics if 'metrics' in locals() else {}
                },
                websocket
            )
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print(f"WebSocket client disconnected during analysis.")
    except Exception as e:
        print(f"Error during video analysis: {e}")
        await manager.send_personal_message({"error": f"Server error during analysis: {e}"}, websocket)
    finally:
        # Check if cap was assigned before trying to release
        if cap is not None:
            cap.release()
        
        # If pose_instance was local to the function, you'd do:
        if pose_instance is not None:
            pose_instance.close()
        
        if os.path.exists(video_path):
            os.remove(video_path)
        manager.disconnect(websocket)
        print(f"Analysis for {filename} finished.")

# --- 運動力學特徵計算範例 ---
def get_landmark_vector(landmark, idx):
    return np.array([landmark[idx].x, landmark[idx].y, landmark[idx].z])


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calc_stride_angle(lm):
    return calculate_angle(
        get_landmark_vector(lm, 24),
        get_landmark_vector(lm, 26),
        get_landmark_vector(lm, 23),
    )


def calc_throwing_angle(lm):
    return calculate_angle(
        get_landmark_vector(lm, 12),
        get_landmark_vector(lm, 14),
        get_landmark_vector(lm, 16),
    )


def calc_arm_symmetry(lm):
    return 1 - abs(lm[15].y - lm[16].y)


def calc_hip_rotation(lm):
    return abs(lm[23].z - lm[24].z)


def calc_elbow_height(lm):
    return lm[14].y


def calc_ankle_height(lm):
    return lm[28].y


def calc_shoulder_rotation(lm):
    return abs(lm[11].z - lm[12].z)


def calc_torso_tilt_angle(lm):
    return calculate_angle(
        get_landmark_vector(lm, 11),
        get_landmark_vector(lm, 23),
        get_landmark_vector(lm, 24),
    )


def calc_release_distance(lm):
    return np.linalg.norm(
        get_landmark_vector(lm, 16) - get_landmark_vector(lm, 12)
    )  # ✅ 修正 list 相減


def calc_shoulder_to_hip(lm):
    return abs(lm[12].x - lm[24].x)

def calculate_pitcher_metrics(landmarks_data: list) -> dict:

    # Convert list of landmarks to dict by id for easier access
    metric_funcs = {
        "stride_angle":       calc_stride_angle,
        "throwing_angle":     calc_throwing_angle,
        "arm_symmetry":       calc_arm_symmetry,
        "hip_rotation":       calc_hip_rotation,
        "elbow_height":       calc_elbow_height,
        "ankle_height":       calc_ankle_height,
        "shoulder_rotation":  calc_shoulder_rotation,
        "torso_tilt_angle":   calc_torso_tilt_angle,
        "release_distance":   calc_release_distance,
        "shoulder_to_hip":    calc_shoulder_to_hip,
    }

    return {
        name: float(round(func(landmarks_data), 2))
        for name, func in metric_funcs.items()
    }

# 運行FastAPI應用 (開發用)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

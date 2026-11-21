from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import cv2
import streamlit as st
from ultralytics import YOLO
import winsound
import os
import smtplib
from email.message import EmailMessage

def send_email_alert(alerts):
    sender = os.getenv("ALERT_EMAIL")
    password = os.getenv("ALERT_PASSWORD")
    receiver = os.getenv("ALERT_TO")

    if not sender or not password or not receiver:
        print("Email settings are missing.")
        return

    msg = EmailMessage()
    msg["Subject"] = "CCTV Security Alert!"
    msg["From"] = sender
    msg["To"] = receiver
    msg.set_content(
        "Dangerous object detected!\n"
        + "\n".join(alerts)
    )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)

        print("Email alert sent successfully!")

    except Exception as e:
        print(f"Email sending failed: {e}")


EVIDENCE_DIR = Path("evidence")
EVIDENCE_DIR.mkdir(exist_ok=True)

DANGEROUS_OBJECTS = {"knife", "scissors"}
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear"}


@st.cache_resource
def load_model(model_name: str) -> YOLO:
    return YOLO(model_name)


def severity(label: str) -> str | None:
    if label in DANGEROUS_OBJECTS:
        return "DANGEROUS OBJECT"
    if label in ANIMALS:
        return "ANIMAL"
    return None


def annotate(frame, results, min_confidence: float):
    alerts: set[str] = set()
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if confidence < min_confidence:
            continue
        label = results[0].names[int(box.cls[0])]
        category = severity(label)
        if not category:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = (0, 0, 255) if category == "DANGEROUS OBJECT" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{category}: {label} {confidence:.0%}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        alerts.add(f"{category}: {label}")
    return frame, sorted(alerts)    


def save_evidence(frame, alerts: list[str]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = EVIDENCE_DIR / f"alert_{stamp}.jpg"
    cv2.imwrite(str(path), frame)
    with (EVIDENCE_DIR / "alerts.log").open("a", encoding="utf-8") as log:
        log.write(f"{datetime.now().isoformat(timespec='seconds')} | {', '.join(alerts)} | {path.name}\n")
    return path


def main() -> None:
    st.set_page_config(page_title="Home CCTV Safety Monitor", page_icon="📹", layout="wide")
    st.title("📹 Home CCTV Safety Monitor")
    st.caption("Detects selected animals and dangerous objects from a webcam, video file, or IP camera stream.")

    with st.sidebar:
        st.header("Settings")
        source = st.radio("Video source", ["Webcam", "Upload video", "IP camera URL"])
        confidence = st.slider("Minimum confidence", 0.10, 0.95, 0.45, 0.05)
        model_name = st.selectbox("Detection model", ["yolo11n.pt", "yolov8n.pt"])
        st.caption("Evidence images and alert history are stored in the `evidence` folder.")

    uploaded_file = None
    stream_url = ""
    if source == "Upload video":
        uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
    elif source == "IP camera URL":
        stream_url = st.text_input("RTSP/HTTP camera URL", placeholder="rtsp://user:password@192.168.1.20:554/stream")

    start = st.button("Start monitoring", type="primary")
    if not start:
        st.info("Select a source and press Start monitoring.")
        return

    if source == "Upload video" and uploaded_file is None:
        st.error("Upload a video file first.")
        return
    if source == "IP camera URL" and not stream_url.strip():
        st.error("Enter the IP camera stream URL first.")
        return

    if source == "Webcam":
        capture = cv2.VideoCapture(0)
    elif source == "IP camera URL":
        capture = cv2.VideoCapture(stream_url.strip())
    else:
        temp_video = Path("uploaded_cctv_video")
        temp_video.write_bytes(uploaded_file.getbuffer())
        capture = cv2.VideoCapture(str(temp_video))

    if not capture.isOpened():
        st.error("Could not open this video source. Check the camera connection or URL.")
        return

    model = load_model(model_name)
    frame_view = st.empty()
    alert_view = st.empty()
    status = st.empty()
    last_saved = 0.0
    status.info("Monitoring started. Close this run or refresh the page to stop.")

    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        results = model(frame, conf=confidence, verbose=False)
        marked_frame, alerts = annotate(frame, results, confidence)
        if alerts:
            alert_view.error("ALERT — " + " | ".join(alerts))
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            send_email_alert(alerts)
            
            if time.time() - last_saved >= 10:
                saved = save_evidence(marked_frame, alerts)
                last_saved = time.time()
                status.warning(f"Evidence saved: {saved}")
        else:
            alert_view.success("No configured dangerous objects or animals detected.")
        frame_view.image(cv2.cvtColor(marked_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    capture.release()
    status.info("Monitoring stopped: video ended or camera stream disconnected.")


if __name__ == "__main__":
    main()

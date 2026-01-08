import streamlit as st
import cv2
import tempfile
import time
import datetime
import threading
import smtplib
import os
import av
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ==============================
# Email Config
# ==============================
EMAIL_ADDRESS = 'uptoskillssunidhi@gmail.com'
EMAIL_PASSWORD = 'wnpf blvr vcll evnu' 
RECIPIENTS = ['Chiruchiranth001@gmail.com']

# Load Model once
model = YOLO("yolov8s.pt")

import requests

# ==============================
# Helper Functions
# ==============================
def get_ip_location():
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()
        if data['status'] == 'success':
            return f"{data['city']}, {data['country']} ({data['lat']},{data['lon']})"
        return "Unknown Location"
    except:
        return "Unknown Location"

def send_email_alert(image_path, location_str):
    subject = "🚨 Accident Detected (Live Camera)"
    body = f"""
    An accident was detected in the live camera feed.
    
    📍 Reported Location: {location_str}
    
    Google Maps: https://www.google.com/maps?q={location_str.split('(')[-1].replace(')', '') if '(' in location_str else location_str}
    """

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ", ".join(RECIPIENTS)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with open(image_path, "rb") as img_file:
            image = MIMEBase('application', 'octet-stream')
            image.set_payload(img_file.read())
            encoders.encode_base64(image)
            image.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(image_path)}"')
            msg.attach(image)
            
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("✅ Email sent.")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

def is_overlapping(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

# ==============================
# WebRTC Video Processor
# ==============================
class AccidentDetectionProcessor(VideoTransformerBase):
    def __init__(self):
        self.collision_start = {}
        self.accident_active = False
        self.last_email_time = 0
        self.location_context = "Unknown" # Set externally

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Detection
        results = model(img, conf=0.3, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                if label in ["car", "truck", "bus", "motorbike", "bicycle"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2))
                    
                    # Draw Box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. Logic (Overlap)
        overlap_found = False
        accident_now = False

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if is_overlapping(detections[i], detections[j]):
                    overlap_found = True
                    
                    # Visual: Red Connection
                    cx1 = (detections[i][0] + detections[i][2]) // 2
                    cy1 = (detections[i][1] + detections[i][3]) // 2
                    cx2 = (detections[j][0] + detections[j][2]) // 2
                    cy2 = (detections[j][1] + detections[j][3]) // 2
                    cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)

                    key = f"{i}-{j}"
                    if key not in self.collision_start:
                        self.collision_start[key] = time.time()
                    elif time.time() - self.collision_start[key] > 2.0:
                        accident_now = True
                else:
                    key = f"{i}-{j}"
                    if key in self.collision_start:
                        del self.collision_start[key]

        # 3. Alerting
        if accident_now:
            self.accident_active = True
            cv2.putText(img, "ACCIDENT DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
            
            # Send Email (Throttled)
            if time.time() - self.last_email_time > 10:
                self.last_email_time = time.time()
                
                # Snapshot
                if not os.path.exists("snapshots"):
                     os.makedirs("snapshots")
                snap_path = f"snapshots/live_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(snap_path, img)
                
                # Use the location context passed during init
                threading.Thread(target=send_email_alert, args=(snap_path, self.location_context), daemon=True).start()
                
        elif not overlap_found:
             self.accident_active = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==============================
# Streamlit App UI
# ==============================
def main():
    st.set_page_config(page_title="Accident Detector Live", page_icon="🚨")
    st.title("🚨 Live Accident Detection")
    st.markdown("Allow camera access to start.")

    # Location Setup
    if 'location' not in st.session_state:
        st.session_state['location'] = get_ip_location()

    with st.sidebar:
        st.header("📍 System Settings")
        st.write(f"Detected IP Location: **{st.session_state['location']}**")
        
        # Allow Manual Override
        manual_loc = st.text_input("Override Location (Optional)", value=st.session_state['location'])
        if manual_loc:
            st.session_state['location'] = manual_loc

    # Video Processor Factory
    # We use a factory to pass the current session location to the processor
    def processor_factory():
        proc = AccidentDetectionProcessor()
        proc.location_context = st.session_state['location']
        return proc

    # Live Camera Stream
    webrtc_streamer(
        key="accident-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=processor_factory,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()

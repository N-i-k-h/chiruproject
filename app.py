import streamlit as st
import cv2
import tempfile
import os
import time
import datetime
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO

# ==============================
# Email Config
# ==============================
EMAIL_ADDRESS = 'uptoskillssunidhi@gmail.com'
EMAIL_PASSWORD = 'wnpf blvr vcll evnu' 
RECIPIENTS = ['Chiruchiranth001@gmail.com']

# Initialize Model
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# ==============================
# Helper Functions
# ==============================
def send_email_alert(image_path):
    subject = "🚨 Accident Detected (Web Upload)"
    body = "An accident was detected in the uploaded video analysis."

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ", ".join(RECIPIENTS)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, "rb") as img_file:
        image = MIMEBase('application', 'octet-stream')
        image.set_payload(img_file.read())
        encoders.encode_base64(image)
        image.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(image_path)}"')
        msg.attach(image)

    try:
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
# Streamlit App
# ==============================
def main():
    st.set_page_config(page_title="Accident Detector", page_icon="🚨")
    
    st.title("🚨 AI Accident Detection System")
    st.markdown("### Upload a traffic video to detect accidents.")

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        
        st_frame = st.empty()
        st_status = st.empty()
        
        accident_active = False
        collision_start = {}
        
        # Create snapshots dir
        if not os.path.exists("snapshots"):
            os.makedirs("snapshots")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 360))
            
            results = model(frame, conf=0.3, verbose=False)
            detections = []
            
            # Draw and Detect
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    if label in ["car", "truck", "bus", "motorbike", "bicycle"]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append((x1, y1, x2, y2))
                        
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Check Overlaps
            overlap_found = False
            accident_now = False

            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    if is_overlapping(detections[i], detections[j]):
                        overlap_found = True
                        
                        # Visual: Red Line
                        cx1 = (detections[i][0] + detections[i][2]) // 2
                        cy1 = (detections[i][1] + detections[i][3]) // 2
                        cx2 = (detections[j][0] + detections[j][2]) // 2
                        cy2 = (detections[j][1] + detections[j][3]) // 2
                        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)

                        key = f"{i}-{j}"
                        if key not in collision_start:
                            collision_start[key] = time.time()
                        elif time.time() - collision_start[key] > 1.5:  # Faster detection for web demo
                            accident_now = True
                    else:
                        key = f"{i}-{j}"
                        if key in collision_start:
                            del collision_start[key]

            if accident_now and not accident_active:
                accident_active = True
                st_status.error("🚨 ACCIDENT DETECTED! Sending Alert...")
                
                # Snapshot
                snap_path = f"snapshots/accident_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(snap_path, frame)
                
                # Email Thread
                threading.Thread(target=send_email_alert, args=(snap_path,), daemon=True).start()

            if not overlap_found:
                accident_active = False
                st_status.info("✅ Monitoring Traffic...")

            if accident_active:
                 cv2.putText(frame, "ACCIDENT DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame, channels="RGB")

        cap.release()
        st.success("Analysis Complete")

if __name__ == "__main__":
    main()

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from ultralytics import YOLO
from math import atan2
import tempfile
import os
import zipfile
from PIL import Image
from datetime import timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from tqdm import tqdm  # Progress bar library

# Load models
MODEL_PATH = "cheating_detection_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)
yolo_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Email sender config
from_email = "sahilmadan0508@gmail.com"        # 🔁 Replace with your email
from_password = "pmnu rqdj sfns mvdl"  # 🔁 Replace with app password

# Helper functions
def normalize_pose(keypoints):
    left_shoulder = keypoints[11 * 3: (11 * 3) + 3]
    right_shoulder = keypoints[12 * 3: (12 * 3) + 3]
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angle = -atan2(dy, dx)

    keypoints_rotated = []
    for i in range(0, len(keypoints), 3):
        x, y, z = keypoints[i], keypoints[i+1], keypoints[i+2]
        x_new = x * np.cos(angle) - y * np.sin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)
        keypoints_rotated.extend([x_new, y_new, z])

    return np.array(keypoints_rotated)

def detect_persons(image):
    results = yolo_model(image)
    detections = results[0].boxes
    persons = []

    for box, conf, cls in zip(detections.xyxy.cpu().numpy(), detections.conf.cpu().numpy(), detections.cls.cpu().numpy()):
        if int(cls) == 0 and conf > 0.35:
            xmin, ymin, xmax, ymax = map(int, box)
            persons.append((xmin, ymin, xmax, ymax))

    return persons

def analyze_frame(frame, detection_threshold=0.5):
    cheating_detected = False
    persons = detect_persons(frame)

    for (xmin, ymin, xmax, ymax) in persons:
        person_crop = frame[ymin:ymax, xmin:xmax]
        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = pose.process(person_crop_rgb)
        if not results.pose_landmarks:
            continue

        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])

        keypoints = normalize_pose(keypoints)
        keypoints = keypoints / np.max(np.abs(keypoints))
        X_test = np.array([keypoints])[:, :, np.newaxis]

        y_pred_prob = model.predict(X_test)[0][0]
        y_pred = 1 if y_pred_prob > detection_threshold else 0

        color = (0, 0, 255) if y_pred == 1 else (0, 255, 0)
        label = "Cheating" if y_pred == 1 else "Not Cheating"
        if y_pred == 1:
            cheating_detected = True

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f"{label} {y_pred_prob*100:.2f}%", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame, cheating_detected

def send_email_alert(to_email, subject="Cheating Alert", body="Cheating has been detected in the uploaded video/image.", attachment=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach ZIP file
        if attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(open(attachment, 'rb').read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment)}')
            msg.attach(part)

        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print("Email error:", e)
        return False

# Streamlit UI
st.set_page_config(page_title="Cheating Detection", page_icon="📸", layout="wide")
st.markdown("""
    <style>
        body { background-color: #121212; color: #fff; font-family: "Helvetica Neue", sans-serif; }
        .stButton > button { background-color: #0078d4; color: white; font-weight: bold; padding: 15px 32px; border-radius: 8px; }
        .stButton > button:hover { background-color: #005a8d; }
        .stFileUploader { background-color: #1a1a1a; padding: 10px; border-radius: 8px; }
        .stTextInput, .stFileUploader { border-radius: 8px; }
        .stMarkdown, .stWrite { color: #bbb; }
    </style>
""", unsafe_allow_html=True)

st.title("🎥 **Cheating Detection from Video or Image**")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], label_visibility="collapsed")
uploaded_image = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
email = st.text_input("📧 Enter your email to receive alerts", placeholder="example@gmail.com")
detection_threshold = st.slider("Cheating Detection Threshold", 0.0, 1.0, 0.5, 0.05)  # Slider for adjusting detection threshold

if uploaded_video is not None:
    temp_dir = tempfile.mkdtemp()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0
    gap = 30  # every 30th frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    saved_frames = []

    stframe = st.empty()
    st.write("⏳ **Processing video...**")

    # Progress bar
    with st.spinner("Processing... Please wait."):
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Frames Processed", position=0, dynamic_ncols=True)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % gap == 0:
                processed_frame, cheating = analyze_frame(frame, detection_threshold)
                timestamp = str(timedelta(seconds=int(frame_count / fps)))

                if cheating:
                    preview_path = os.path.join(temp_dir, f"cheat_frame_{frame_count}.jpg")
                    cv2.imwrite(preview_path, processed_frame)
                    saved_frames.append((preview_path, timestamp))

                stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count}", channels="RGB")

            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()

    if saved_frames:
        st.success(f"⚠️ Detected cheating in {len(saved_frames)} frame(s).")
        st.write("### 🖼️ **Preview of Detected Cheating Frames**:")
        for path, timestamp in saved_frames:
            st.image(path, caption=f"Cheating detected at {timestamp}", use_column_width=True)

        # Email alert
        if email:
            # Create a ZIP file of the detected frames
            zip_path = os.path.join(temp_dir, "cheating_frames.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for img_path, _ in saved_frames:
                    zipf.write(img_path, os.path.basename(img_path))

            if send_email_alert(email, subject="Cheating Detection Completed", body=f"Cheating has been detected in {len(saved_frames)} frames.", attachment=zip_path):
                st.success(f"📨 Email alert sent to {email} with ZIP attachment.")
            else:
                st.error("⚠️ Failed to send email. Please check the address or credentials.")

        # ZIP download
        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ **Download Cheating Frames (ZIP)**",
                data=f,
                file_name="cheating_frames.zip",
                mime="application/zip"
            )
    else:
        st.info("✅ **No cheating detected in sampled frames.**")

elif uploaded_image is not None:
    img_pil = Image.open(uploaded_image).convert("RGB")
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    processed_img, cheating = analyze_frame(img_bgr, detection_threshold)

    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

    if cheating:
        st.success("⚠️ **Cheating detected in the image!**")

        if email:
            if send_email_alert(email, attachment=None):
                st.success(f"📨 Email alert sent to {email}")
            else:
                st.error("⚠️ Failed to send email. Please check the address or credentials.")

        preview_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(preview_path.name, processed_img)

        with open(preview_path.name, "rb") as f:
            st.download_button(
                label="⬇️ **Download Image with Detected Cheating**",
                data=f,
                file_name="cheating_detected_image.jpg",
                mime="image/jpeg"
            )
    else:
        st.info("✅ **No cheating detected in the image.**")

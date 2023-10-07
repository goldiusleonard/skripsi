from typing import Optional
import streamlit as st
import io
import numpy as np
import gc
import cv2
from face_anti_spoofing.main import SpoofDetection
from face_detection import RetinaFace
from util.files import get_allowed_types

gc.enable()

COLOR_RED = (0, 0, 153)
COLOR_GREEN = (0, 153, 0)

FRAME_WINDOW = st.image([])

CURRENT_FOLDER = "skripsi_demo"

def check_webcam():
    webcam_dict = dict()
    for i in range(0, 10):
        cap = cv2.VideoCapture(i)
        is_camera = cap.isOpened()
        if is_camera:
            webcam_dict[f"index[{i}]"] = "VALID"
            cap.release()
        else:
            webcam_dict[f"index[{i}]"] = None
            cap.release()
    return webcam_dict

def recognize_spoof(frame, dets, threshold, model, sd_model, cam_frame: Optional = None):
    dets = dets.astype(np.int)
    
    cropped_frame = frame[dets[1]:dets[3], dets[0]:dets[2]]
    
    if "mining" not in model:
        preprocessed_img = sd_model.preprocess_image(cropped_frame.copy())
    else:
        preprocessed_img = sd_model.preprocess_image_alter(cropped_frame.copy())

    preds = sd_model.predict(preprocessed_img)

    label = "-"
    if preds > threshold:
        label = f"real"
        COLOR = COLOR_GREEN
    else:
        label = f"fake"
        COLOR = COLOR_RED
    if cam_frame is not None:
        if COLOR == COLOR_RED:
            COLOR = (153, 0, 0)
        cv2.rectangle(cam_frame, (dets[0], dets[1]), (dets[2], dets[3]), COLOR, 2)
        cv2.putText(cam_frame, label, (dets[0], dets[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR, 2)
    else:
        cv2.rectangle(frame, (dets[0], dets[1]), (dets[2], dets[3]), COLOR, 2)
        cv2.putText(frame, label, (dets[0], dets[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR, 2)
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
    st.title("Face Anti-Spoofing Skripsi Demo")
    input_method = st.radio("Input method", ["Upload File", "Webcam"], index=0,label_visibility="visible")

    choices = []
    thresholds = []

    f = open("./face_anti_spoofing/weights/modelandthreshold.txt", "r")
    for line in f:
        choices.append(line.split(':')[0])
        thresholds.append(line.split(':')[1])

    choice = st.selectbox("Select model",choices)
    threshold = float(thresholds[choices.index(choice)])
    
    sd = SpoofDetection(choice)
    fd = RetinaFace()
    video_capture = None
    
    if input_method == "Upload File":

        with st.form("form", clear_on_submit=True):
            allowed_exts = get_allowed_types()

            file = st.file_uploader(
                "Upload image", accept_multiple_files=False, type=allowed_exts
            )
            submitted = st.form_submit_button("Submit")

            if submitted and file is not None:
                buffer = io.BytesIO(file.read())
                img_array = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                faces = fd(img)
                
                if len(faces) <= 0:
                    raise RuntimeWarning("No Faces Detected!")
                else:
                    box, landmarks, score = faces[0]
                    recognize_spoof(img, box, threshold, choice, sd)
            gc.collect()
    elif input_method == "Webcam":
        webcam_dict = check_webcam()
        for num, i in enumerate(webcam_dict):
            if webcam_dict[i] != 'VALID' and num == 0:
                st.write("no webcam detected!")
                break
            if webcam_dict[i] == 'VALID':
                WEBCAMNUM = num
        
        video_capture = cv2.VideoCapture(WEBCAMNUM)
        
        while(True):
            label = "-"
            ret, frame = video_capture.read()
            if ret:
                FRAME_WINDOW.image(cv2.cvtColor(frame[:, :, ::-1], cv2.COLOR_BGR2RGB))
                faces = fd(frame[:, :, ::-1])
                if len(faces) <= 0:
                    continue
                else:
                    box, landmarks, score = faces[0]
                    recognize_spoof(frame[:, :, ::-1], box, threshold, choice, sd, frame)
            gc.collect()
    
    if input_method != "Webcam" and video_capture is not None:
        video_capture.release()
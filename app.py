import time

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Load pre-prepared video
video_path = "video/instructor1.mp4"
frame_placeholder = st.empty()

@st.cache(allow_output_mutation=True)
def load_model():
    # MediaPipe pose estimation initialization
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    return pose, mp_drawing


pose, mp_drawing = load_model()


def get_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]
        return np.array(landmarks).flatten(), results.pose_landmarks
    return [], None


def pose_similarity_coefficient(landmarks1, landmarks2):
    if landmarks1 == [] or landmarks2 == []:
        print("Landmarks not detected in one or both images")
        return 0

    dot_product = np.dot(landmarks1, landmarks2)
    norm_landmarks1 = np.linalg.norm(landmarks1)
    norm_landmarks2 = np.linalg.norm(landmarks2)

    cosine_similarity = dot_product / (norm_landmarks1 * norm_landmarks2)

    # Adjusting to the range 0-1
    probability = (cosine_similarity + 1) / 2
    return probability


def visualize_frames(frame1, frame2, landmarks1, landmarks2, similarity):
    # Draw skeletons on the frames
    if landmarks1 is not None:
        mp_drawing.draw_landmarks(frame1, landmarks1, mp_pose.POSE_CONNECTIONS)

    # Resize both frames to a common size
    target_height = min(frame1.shape[0], frame2.shape[0])
    target_width = min(frame1.shape[1], frame2.shape[1])

    frame1_resized = cv2.resize(frame1, (target_width, target_height))
    frame2_resized = cv2.resize(frame2, (target_width, target_height))

    # Now stack the two frames
    combined_frame = np.hstack((frame1_resized, frame2_resized))

    if similarity > 0.97:
        # Add the similarity text on the frame
        cv2.putText(combined_frame, f"Great Job! Similarity: {similarity*100:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0),
                    2)
    else:
        # Add the similarity text on the frame
        cv2.putText(combined_frame, f"You are doing it wrong", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255),
                    2)

    return combined_frame


st.title("Video Pose Alba Yoga Similarity Demo")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.start_time = time.time()  # Set the timestamp when the instance is created
        self.cap_video = cv2.VideoCapture(video_path)

    def transform(self, frame):
        # Convert frame to grayscale
        img = frame.to_ndarray(format="bgr24")

        elapsed_time = time.time() - self.start_time

        if elapsed_time < 10:  # 120 seconds = 2 minutes
            return img

        # Use your previous code here to process the `img` and return the result
        ret_video, frame_video = self.cap_video.read()

        if not ret_video:
            return img

        landmarks_cam, pose_landmarks1 = get_landmarks(img)
        landmarks_video, pose_landmarks2 = get_landmarks(frame_video)

        if landmarks_cam != [] or landmarks_video != []:
            similarity = pose_similarity_coefficient(landmarks_cam, landmarks_video)
            img = visualize_frames(img, frame_video, pose_landmarks1, pose_landmarks2, similarity)

        return img

    def on_ended(self):
        self.cap_video.release()


webrtc_streamer(key="live", video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},)


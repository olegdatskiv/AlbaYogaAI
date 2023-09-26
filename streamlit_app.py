import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from time import sleep


# Initialize session state for video playing
if 'playing' not in st.session_state:
    st.session_state.playing = False

# Load pre-prepared video
video_path = "video/instructor1.mp4"
cap_video = cv2.VideoCapture(video_path)
frame_placeholder = st.empty()

# MediaPipe pose estimation initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


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


st.title("Video Pose Similarity")

if st.session_state.playing:
    st.write('Playing video...')

    # Stop button
    if st.button('Stop'):
        st.session_state.playing = False

    # Capture from camera
    camera = cv2.VideoCapture(1)
    for i in range(300):  # 10 seconds at 30fps
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to grab frame from camera.")
            break
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        sleep(0.03)  # ~30fps

    # Play pre-prepared video and compare
    while cap_video.isOpened():
        ret_video, frame_video = cap_video.read()
        ret_cam, frame_cam = camera.read()

        if not ret_video or not ret_cam:
            break

        landmarks_cam, pose_landmarks1 = get_landmarks(frame_cam)
        landmarks_video, pose_landmarks2 = get_landmarks(frame_video)

        if landmarks_cam != [] or landmarks_video != []:
            similarity = pose_similarity_coefficient(landmarks_cam, landmarks_video)
            frame_cam = visualize_frames(frame_cam, frame_video, pose_landmarks1, pose_landmarks2, similarity)

        frame_placeholder.image(frame_cam, channels="BGR", use_column_width=True)
        sleep(0.03)  # ~30fps

    camera.release()
    cap_video.release()

    # Reset playing state once done
    st.session_state.playing = False

else:
    # Here you can put any default display or instructions when the video isn't playing
    st.write("Press Start to begin!")

    if st.button("Start"):
        st.session_state.playing = True

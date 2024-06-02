import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe and OpenCV modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to count curls using pose estimation
def count_curls(video_path=0):
    cap = cv2.VideoCapture(video_path)
    frame_window = st.image([])  # Initialize Streamlit image display

    # Curl counter variables
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image color from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make pose detection
            results = pose.process(image)

            # Convert the image color back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and calculate angles
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(counter)

            # Render curl counter and status
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Update Streamlit image display
            frame_window.image(image, channels='BGR')



        cap.release()


# Function to count sit-ups using pose estimation
def count_situps(video_path=0):
    cap = cv2.VideoCapture(video_path)
    frame_window = st.image([])

    # Sit-up counter variables
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image color from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make pose detection
            results = pose.process(image)

            # Convert the image color back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and calculate angles
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle = calculate_angle(hip, knee, ankle)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Sit-up counter logic
                if angle > 160:
                    stage = "down"
                if angle < 60 and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(counter)

            # Render sit-up counter and status
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            frame_window.image(image, channels='BGR')

        cap.release()


# Function to count push-ups using pose estimation
def count_pushups(video_path=0):
    cap = cv2.VideoCapture(video_path)
    frame_window = st.image([])

    # Push-up counter variables
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image color from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make pose detection
            results = pose.process(image)

            # Convert the image color back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and calculate angles
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Push-up counter logic
                if angle > 160:
                    stage = "up"
                if angle < 60 and stage == 'up':
                    stage = "down"
                    counter += 1
                    print(counter)

            # Render push-up counter and status
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            frame_window.image(image, channels='BGR')

        cap.release()


# Function to count squats using pose estimation
def count_squats(video_path=0):
    cap = cv2.VideoCapture(video_path)
    frame_window = st.image([])

    # Squat counter variables
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image color from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make pose detection
            results = pose.process(image)

            # Convert the image color back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and calculate angles
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle = calculate_angle(hip, knee, ankle)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Squat counter logic
                if angle > 160:
                    stage = "up"
                if angle < 60 and stage == 'up':
                    stage = "down"
                    counter += 1
                    print(counter)

            # Render squat counter and status
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            frame_window.image(image, channels='BGR')

        cap.release()


# Function to count lunges using pose estimation
def count_lunges(video_path=0):
    cap = cv2.VideoCapture(video_path)
    frame_window = st.image([])

    # Lunge counter variables
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image color from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make pose detection
            results = pose.process(image)

            # Convert the image color back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and calculate angles
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle = calculate_angle(hip, knee, ankle)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Lunge counter logic
                if angle > 160:
                    stage = "up"
                if angle < 60 and stage == 'up':
                    stage = "down"
                    counter += 1
                    print(counter)

            # Render lunge counter and status
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            frame_window.image(image, channels='BGR')

        cap.release()


def main():
    st.title("Workout Counter Application")
    st.image("coverpage.png")
    st.markdown("## Welcome to the Workout Counter Application")
    st.markdown(
        """
        This application allows you to count various workouts in real-time or from a video file. 
        Select a workout from the dropdown menu, choose whether to use your webcam or upload a video, and start counting!

        The Workout Counter Application is designed to help fitness enthusiasts, personal trainers, and health-conscious individuals track their exercise routines more efficiently. By leveraging computer vision and machine learning technologies, the application provides real-time feedback and accurate counts of workout repetitions. This can be particularly beneficial for maintaining proper form, maximizing workout effectiveness, and reducing the risk of injury.

        Our application supports various workouts including curls, sit-ups, push-ups, squats, and lunges. Each workout is tracked using advanced pose estimation techniques, ensuring that every repetition is counted accurately. Users can select their preferred workout from a simple dropdown menu and choose to use either a live webcam feed or upload a pre-recorded video for analysis.

        In addition to tracking workouts, the application offers recommendations and benefits for each exercise, helping users understand the impact of their routines on different muscle groups. We also provide links to high-quality YouTube videos for each workout, offering guidance on proper techniques and variations to enhance your fitness journey.

        The Workout Counter Application is user-friendly and easy to navigate. Upon logging in, users are greeted with a clean interface where they can start their workout session. Real-time feedback is provided on the screen, displaying the current count of repetitions and the user's form status. This feature helps users correct their form on the fly, ensuring they get the most out of their exercise.

        Whether you are a beginner looking to get started on your fitness journey or an experienced athlete seeking to optimize your workouts, the Workout Counter Application is here to support your goals. Our goal is to make fitness accessible and enjoyable for everyone, providing the tools needed to achieve a healthier, stronger, and more active lifestyle.
        """
    )

    # User login section in sidebar
    st.sidebar.title("Login")
    if 'login' not in st.session_state:
        st.session_state['login'] = False

    if not st.session_state['login']:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username == "admin" and password == "password":  # Simple check
                st.session_state['login'] = True
                st.sidebar.success("Logged in successfully!")
            else:
                st.sidebar.error("Invalid username or password")
    else:
        st.sidebar.success("Logged in as admin")

        # Workout selection
        workout = st.selectbox(
            "Select a Workout",
            ("Count Curls", "Count Push-ups", "Count Squats", "Count Lunges")
        )

        video_source = st.radio("Choose Video Source", ["Live Video"])

        video_path = 0  # Use webcam

        if st.button("Start Workout"):
            if workout == "Count Curls":
                count_curls(video_path)
            elif workout == "Count Sit-ups":
                count_situps(video_path)
            elif workout == "Count Push-ups":
                count_pushups(video_path)
            elif workout == "Count Squats":
                count_squats(video_path)
            elif workout == "Count Lunges":
                count_lunges(video_path)

        st.markdown("### Workout Recommendations and Benefits")

        # Display slider to select number of videos
        num_videos = st.slider("Select number of videos to display", min_value=1, max_value=5, value=3)

        # Initialize session state for workout and diet recommendations
        if 'num_videos' not in st.session_state:
            st.session_state['num_videos'] = num_videos

        # Update session state with the current number of videos
        st.session_state['num_videos'] = num_videos

        # Display all workout and diet recommendations
        workouts = {
            "Curls Workout": [
                "ykJmrZ5v0Oo",
                "JyV7mUFSpXs",
                "B0HaGjZW5Cc",
                "jDvdHXpA1Eg",
                "vFXwQSuY_gw"
            ],
            "Sit-ups Workout": [
                "jDwoBqPH0jk",
                "1fbU_MkV7NE",
                "z6PJMT2y8GQ",
                "onaQ0v_J5uU",
                "A7Y2-G4zOUA"
            ],
            "Push-ups Workout": [
                "_l3ySVKYVJ8",
                "IODxDxX7oi4",
                "n69-eVLtevc",
                "b5e8QwnKAP0",
                "fUk-rdHDl3w"
            ],
            "Squats Workout": [
                "gsNoPYwWXeM",
                "YaXPRqUwItQ",
                "QrySowAxVrM",
                "IB_icWRzi4E",
                "4KmY44Xsg2w"
            ],
            "Lunges Workout": [
                "MxfTNXSFiYI",
                "3XDriUn0udo",
                "1LuRcKJMn8w",
                "wrwwXE_x-pQ",
                "Q_Bpj91Yiis"
            ],
        }

        diets = {
            "High Protein Diet": [
                "XBvMt45d66Q",
                "pRlpPOj6HnY",
                "j61CcVYlCas",
                "n9wQrjPlgc0",
                "3feSPKetkmI"
            ],
            "Keto Diet": [
                "2-dovKW9bjo",
                "xVqigEvE4p4",
                "0V2iJNWX6i0",
                "9h9S9kD67-Q",
                "rQblUItn-m8"
            ],
        }

        if st.button("View Workout and Diet Recommendations"):
            st.markdown("#### Workouts:")
            for workout_name, video_ids in workouts.items():
                st.title(f"**{workout_name}**")
                for video_id in video_ids[:st.session_state['num_videos']]:
                    thumb_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
                    st.image(thumb_url, caption=workout_name, use_column_width=True)
                    st.markdown(f"[Watch Video](https://www.youtube.com/watch?v={video_id})")

            st.title("#### Diets:")
            for diet_name, video_ids in diets.items():
                st.title(f"**{diet_name}**")
                for video_id in video_ids[:st.session_state['num_videos']]:
                    thumb_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
                    st.image(thumb_url, caption=diet_name, use_column_width=True)
                    st.markdown(f"[Watch Video](https://www.youtube.com/watch?v={video_id})")

st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
        <p>Developed by Vignesh L - &copy; 2024</p>
    </div>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()

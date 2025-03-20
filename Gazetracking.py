import mediapipe as mp
import cv2
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
draw_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


roi_x1, roi_y1 = 100, 100  
roi_x2, roi_y2 = 300, 300 
EYE_INDICES = [33, 133, 160, 144, 145, 153, 154, 155]  

def get_average_eye_position(face_landmarks, image_shape):
    eye_points = [face_landmarks.landmark[i] for i in EYE_INDICES]
    x_coords = [int(point.x * image_shape[1]) for point in eye_points]
    y_coords = [int(point.y * image_shape[0]) for point in eye_points]
    avg_x = sum(x_coords) // len(x_coords)
    avg_y = sum(y_coords) // len(y_coords)
    return avg_x, avg_y

def is_looking_at_roi(face_landmarks, image_shape):
    avg_x, avg_y = get_average_eye_position(face_landmarks, image_shape)
    return roi_x1 <= avg_x <= roi_x2 and roi_y1 <= avg_y <= roi_y2

def draw_landmarks(image , results):
        image.flags.writeable = True
        if results.multi_face_landmarks:
                for face_landmark in results.multi_face_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                                image= image,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_list= face_landmark,
                                landmark_drawing_spec= None,
                                connection_drawing_spec= mp_drawing_styles.get_default_face_mesh_tesselation_style()
                                )
                        mp.solutions.drawing_utils.draw_landmarks(
                                image=image,
                                landmark_list = face_landmark,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec= None,
                                connection_drawing_spec= mp_drawing_styles.get_default_face_mesh_contours_style()

                        )
                        mp.solutions.drawing_utils.draw_landmarks(
                                image=image,
                                landmark_list = face_landmark,
                                connections=mp_face_mesh.FACEMESH_IRISES,
                                landmark_drawing_spec= None,
                                connection_drawing_spec= mp_drawing_styles.get_default_face_mesh_iris_connections_style()

                        )
        return image               

cap = cv2.VideoCapture(0)
attention_lost_time = None
attention_threshold = 5  

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0] 
            if is_looking_at_roi(face_landmarks, (image_height, image_width)):
                attention_lost_time = None
            else:
                if attention_lost_time is None:
                    attention_lost_time = time.time()
                elif time.time() - attention_lost_time > attention_threshold:
                    print("Hey, focus on reading!")
                    cv2.putText(image, 'Focus on reading!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            draw_landmarks(image, results)

        cv2.imshow('FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == 27:  
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

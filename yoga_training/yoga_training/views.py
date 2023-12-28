from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.http.response import StreamingHttpResponse
import numpy as np
import tensorflow as tf 
import mediapipe as mp 
from django.views.decorators.csrf import csrf_exempt
import cv2
import pandas as pd
import threading
from django.middleware import csrf
import json
import os

# from tensorflowjs.converters import convert_tf_saved_model

# model = tf.keras.models.load_model('pose_classification_model.h5')

# input_shape = model.inputs[0].shape
# output_shape = model.outputs[0].shape

# convert_tf_saved_model('pose_classification_model.h5', '/output_directory',
#                         input_shapes={'input_layer_name': input_shape},
#                         output_names=['output_layer_name'])




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

poses={
    0:'no_pose',
    1:'tree',
    2:'chair',
    3:'cobra',
    4:'shoulder_stand',
    5:'warrior',
    6:'triangle',
    
}

# current_directory = os.getcwd()  # Get the current working directory
# files_in_directory = os.listdir(current_directory)
# print(files_in_directory)
classification_modal=tf.keras.models.load_model('pose_classification_model.h5')

class PoseDetectionThread(threading.Thread):
    def __init__(self, image):
        threading.Thread.__init__(self)
        self.image = image
        self.result = None

    def run(self):
        # Perform pose detection
        self.result = PoseDetion(self.image)

    def get_pose(self):
        # Start the thread and then wait for it to complete
        self.start()
        self.join()
        return self.result



class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.


		# ret, jpeg = cv2.imencode('.jpg', image)
		return PoseDetectionThread(image).get_pose()


def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
		

def Video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')

def Home(request):
    # Your view logic goes here
    return render(request, 'home.html', {'message': 'hi'})


def Pose(request,data_id):
    images=['','tree.jpg','chair.jpg','cobra.jpg','shoulder_stand.jpg','warrior.jpg','traingle.jpg']
    # Your view logic goes here
    return render(request, 'pose.html', {'image': images[data_id]})

@csrf_exempt
def Pose_stream(request):
    if request.method == 'POST':
        try:
            frame_data = request.POST.get('frame')
            
            # Process the frame_data as needed
            # For simplicity, let's echo the received frame back
            return JsonResponse({'frame': frame_data})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return HttpResponse("Only POST requests are allowed.", status=400)


def find_center(keypoints):
    # Calculate the center by taking the average of x and y coordinates
    center_x = np.mean([x for x, y in keypoints])
    center_y = np.mean([y for x, y in keypoints])
    return center_x, center_y

def normalize_distances(distances):
    max_distance = max(distances)
    min_distance = min(distances)
    
    if max_distance == min_distance:
        # Avoid division by zero if all distances are the same
        return [0.0] * len(distances)
    
    return [(distance - min_distance) / (max_distance - min_distance) for distance in distances]



def calculate_distances_and_angles(keypoints, center):
    distances = []
    angles = []

    for x, y in keypoints:
        # Calculate Euclidean distance from the center
        distance = np.linalg.norm(np.array([x, y]) - np.array(center))
        distances.append(distance)

        # Calculate angle relative to the center (you may need to adjust this)
        angle = (np.arctan2(y - center[1], x - center[0]) * 180.0 / np.pi + 360) % 360
        angles.append(angle)
    
    normalized_distances = normalize_distances(distances)
    
    return normalized_distances, angles

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
#     if angle >180.0:
#         angle = 360-angle
    
    # Round to 1 decimal place
    angle = round(angle, 1)
    
        
    return angle 




def PoseDetion(image):
        
        image = cv2.resize(image, (640, 480))

        
        # Recolor image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5).process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
                    # Extract key landmarks for angle calculation
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                          landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
            keypoints=[
                right_shoulder,right_elbow,right_wrist,right_hip,right_knee,right_ankle,
                left_shoulder,left_elbow,left_wrist,left_hip,left_knee,left_ankle
            ]
            
            #finding center
            center = nose


            #find normalized distances and angles
            normalized_distance,angles_from_center=calculate_distances_and_angles(keypoints,center)
            
    
        # Calculate angles
            angle_right_shoulder_hip_knee = calculate_angle(right_shoulder, right_hip, right_knee)
            angle_right_hip_knee_ankle = calculate_angle(right_hip, right_knee, right_ankle)
            angle_left_shoulder_hip_knee = calculate_angle(left_shoulder, left_hip, left_knee)
            angle_left_hip_knee_ankle = calculate_angle(left_hip, left_knee, left_ankle)
        
            angle_right_elbow_shoulder_hip = calculate_angle(right_elbow, right_shoulder, right_hip)
            angle_left_elbow_shoulder_hip = calculate_angle(left_elbow, left_shoulder, left_hip)
        
            angle_right_wrist_elbow_shoulder = calculate_angle(right_wrist,right_elbow,right_shoulder)
            angle_left_wrist_elbow_shoulder = calculate_angle(left_wrist,left_elbow,left_shoulder)
        
            
            # # Visualize angle
            
            # #elbow
            # cv2.putText(image, str(angle_left_wrist_elbow_shoulder)+" , "+str(round(normalized_distance[7],2))+" , "+str(round(angles_from_center[7],2)), 
            #                tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )            
            # cv2.putText(image, str(angle_right_wrist_elbow_shoulder)+" , "+str(round(normalized_distance[1],2))+" , "+str(round(angles_from_center[1],2)), 
            #                tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            
            # #hip
            # cv2.putText(image, str(angle_right_shoulder_hip_knee)+" , "+str(round(normalized_distance[3],2))+" , "+str(round(angles_from_center[3],2)), 
            #                tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image, str(angle_left_shoulder_hip_knee)+" , "+str(round(normalized_distance[9],2))+" , "+str(round(angles_from_center[9],2)), 
            #                tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            
            # #knee
            # cv2.putText(image, str(angle_right_hip_knee_ankle)+" , "+str(round(normalized_distance[4],2))+" , "+str(round(angles_from_center[4],2)), 
            #                tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image, str(angle_left_hip_knee_ankle)+" , "+str(round(normalized_distance[10],2))+" , "+str(round(angles_from_center[10],2)), 
            #                tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            
            # #shoulder
            # cv2.putText(image, str(angle_right_elbow_shoulder_hip)+" , "+str(round(normalized_distance[0],2))+" , "+str(round(angles_from_center[0],2)), 
            #                tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image, str(angle_left_elbow_shoulder_hip)+" , "+str(round(normalized_distance[6],2))+" , "+str(round(angles_from_center[6],2)), 
            #                tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
                        
            # #wrist
            # cv2.putText(image," , "+str(round(normalized_distance[2],2))+" , "+str(round(angles_from_center[2],2)), 
            #                tuple(np.multiply(right_wrist, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image," , "+str(round(normalized_distance[8],2))+" , "+str(round(angles_from_center[8],2)), 
            #                tuple(np.multiply(left_wrist, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # #ankle
            # cv2.putText(image," , "+str(round(normalized_distance[5],2))+" , "+str(round(angles_from_center[5],2)), 
            #                tuple(np.multiply(right_ankle, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # cv2.putText(image," , "+str(round(normalized_distance[11],2))+" , "+str(round(angles_from_center[11],2)), 
            #                tuple(np.multiply(left_ankle, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            # center_x, center_y = tuple(np.multiply(center, [640, 480]).astype(int))
            # cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

            
            data={
    'angle_right_shoulder_hip_knee': angle_right_shoulder_hip_knee,
    'angle_right_hip_knee_ankle': angle_right_hip_knee_ankle,
    'angle_left_shoulder_hip_knee':angle_left_shoulder_hip_knee ,
    'angle_left_hip_knee_ankle': angle_left_hip_knee_ankle,
    'angle_right_elbow_shoulder_hip': angle_right_elbow_shoulder_hip,
    'angle_left_elbow_shoulder_hip': angle_left_elbow_shoulder_hip,
    'angle_right_wrist_elbow_shoulder': angle_right_wrist_elbow_shoulder,
    'angle_left_wrist_elbow_shoulder': angle_left_wrist_elbow_shoulder,
    'ndright_shoulder': normalized_distance[0],
    'aright_shoulder': angles_from_center[0],
    'ndright_elbow': normalized_distance[1],
    'aright_elbow': angles_from_center[1],
    'ndright_wrist': normalized_distance[2],
    'aright_wrist': angles_from_center[2],
    'ndright_hip': normalized_distance[3],
    'aright_hip': angles_from_center[3],
    'ndright_knee': normalized_distance[4],
    'aright_knee': angles_from_center[4],
    'ndright_ankle': normalized_distance[5],
    'aright_ankle': angles_from_center[5],
    'ndleft_shoulder': normalized_distance[6],
    'aleft_shoulder': angles_from_center[6],
    'ndleft_elbow': normalized_distance[7],
    'aleft_elbow': angles_from_center[7],
    'ndleft_wrist': normalized_distance[8],
    'aleft_wrist': angles_from_center[8],
    'ndleft_hip': normalized_distance[9],
    'aleft_hip': angles_from_center[9],
    'ndleft_knee': normalized_distance[10],
    'aleft_knee': angles_from_center[10],
    'ndleft_ankle': normalized_distance[11],
    'aleft_ankle': angles_from_center[11],
}

                       
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )  
        

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
        

@csrf_exempt
def Coordinates(request):
    if request.method == 'POST':
        try:
            # Get coordinates from the request data
            raw_data = request.body.decode('utf-8')
            
            # Parse the JSON data
            data = json.loads(raw_data)

            # Access the 'coordinates' key from the parsed data
            keypoint=[]
            coordinates = data
            nose=[coordinates[0]['x'],coordinates[0]['y']]
            right_shoulder=[coordinates[1]['x'],coordinates[1]['y']]
            right_elbow=[coordinates[2]['x'],coordinates[2]['y']]
            right_wrist=[coordinates[3]['x'],coordinates[3]['y']]
            right_hip=[coordinates[4]['x'],coordinates[4]['y']]
            right_knee=[coordinates[5]['x'],coordinates[5]['y']]
            right_ankle=[coordinates[6]['x'],coordinates[6]['y']]
            left_shoulder=[coordinates[7]['x'],coordinates[7]['y']]
            left_elbow=[coordinates[8]['x'],coordinates[8]['y']]
            left_wrist=[coordinates[9]['x'],coordinates[9]['y']]
            left_hip=[coordinates[10]['x'],coordinates[10]['y']]
            left_knee=[coordinates[11]['x'],coordinates[11]['y']]
            left_ankle=[coordinates[12]['x'],coordinates[12]['y']]
            for i in coordinates[1:]:
                 keypoint.append([i["x"], i["y"]])
            
            angle_right_shoulder_hip_knee = calculate_angle(right_shoulder, right_hip, right_knee)
            angle_right_hip_knee_ankle = calculate_angle(right_hip, right_knee, right_ankle)
            angle_left_shoulder_hip_knee = calculate_angle(left_shoulder, left_hip, left_knee)
            angle_left_hip_knee_ankle = calculate_angle(left_hip, left_knee, left_ankle)
        
            angle_right_elbow_shoulder_hip = calculate_angle(right_elbow, right_shoulder, right_hip)
            angle_left_elbow_shoulder_hip = calculate_angle(left_elbow, left_shoulder, left_hip)
        
            angle_right_wrist_elbow_shoulder = calculate_angle(right_wrist,right_elbow,right_shoulder)
            angle_left_wrist_elbow_shoulder = calculate_angle(left_wrist,left_elbow,left_shoulder)

            normalized_distance,angles_from_center=calculate_distances_and_angles(keypoint,nose)



            All_measures={
    'angle_right_shoulder_hip_knee': angle_right_shoulder_hip_knee,
    'angle_right_hip_knee_ankle': angle_right_hip_knee_ankle,
    'angle_left_shoulder_hip_knee':angle_left_shoulder_hip_knee ,
    'angle_left_hip_knee_ankle': angle_left_hip_knee_ankle,
    'angle_right_elbow_shoulder_hip': angle_right_elbow_shoulder_hip,
    'angle_left_elbow_shoulder_hip': angle_left_elbow_shoulder_hip,
    'angle_right_wrist_elbow_shoulder': angle_right_wrist_elbow_shoulder,
    'angle_left_wrist_elbow_shoulder': angle_left_wrist_elbow_shoulder,
    'ndright_shoulder': normalized_distance[0],
    'aright_shoulder': angles_from_center[0],
    'ndright_elbow': normalized_distance[1],
    'aright_elbow': angles_from_center[1],
    'ndright_wrist': normalized_distance[2],
    'aright_wrist': angles_from_center[2],
    'ndright_hip': normalized_distance[3],
    'aright_hip': angles_from_center[3],
    'ndright_knee': normalized_distance[4],
    'aright_knee': angles_from_center[4],
    'ndright_ankle': normalized_distance[5],
    'aright_ankle': angles_from_center[5],
    'ndleft_shoulder': normalized_distance[6],
    'aleft_shoulder': angles_from_center[6],
    'ndleft_elbow': normalized_distance[7],
    'aleft_elbow': angles_from_center[7],
    'ndleft_wrist': normalized_distance[8],
    'aleft_wrist': angles_from_center[8],
    'ndleft_hip': normalized_distance[9],
    'aleft_hip': angles_from_center[9],
    'ndleft_knee': normalized_distance[10],
    'aleft_knee': angles_from_center[10],
    'ndleft_ankle': normalized_distance[11],
    'aleft_ankle': angles_from_center[11],
}
            
            All_measures_df=pd.DataFrame.from_dict(All_measures, orient='index', columns=['value']).transpose()

            predictions = classification_modal.predict(All_measures_df, verbose=0)

# Get the index of the maximum value in the predictions
            predicted_class_index = np.argmax(predictions)

# Map the index to the corresponding pose label
            predicted_pose = poses[predicted_class_index]

            # For demonstration purposes, just echoing the coordinates in the response
            if (predicted_pose=='tree') :
                response_data = {'success': True, 'predicted_pose': predicted_pose}
                return JsonResponse(response_data)
            return JsonResponse({})
        except Exception as e:
            print(f"Exception in Coordinates view: {e}")
            # Handle any exceptions that might occur during processing
            response_data = {'success': False, 'error': str(e)}
            return JsonResponse(response_data, status=500)

    # Return an error response for non-POST requests
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=400)

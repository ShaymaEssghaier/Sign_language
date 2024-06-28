import cv2
import pandas as pd
import random as rd
import mediapipe as mp
import numpy as np
import json
import os

# instanciate models
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def define_keypoints_to_keep():
    '''
    Select only the keypoints we want to keep from keypoints return by modelpipe models Hand, Pose and Facemesh
    
    Paramaters
    ---------
    None

    Returns
    -------
    hands_keypoints, pose_keypoints, face_keypoints: tuple of list of integer (keypoints id)
    '''
    face_keypoints_correspondance_filepath = 'face_keypoints_correspondance.json'

    with open(face_keypoints_correspondance_filepath, 'r') as f:
        face_keypoints_correspondance = json.load(f)
    hands_keypoints = list(range(21))
    pose_keypoints = [11, 12, 13, 14, 15, 16]
    face_keypoints_temp = [ face_keypoints_correspondance[keypoint_area] for keypoint_area in \
                                                                        [#'silhouette',\
                                                                        #'lipsUpperOuter',\
                                                                        #'lipsLowerOuter',\
                                                                        'lipsUpperInner',\
                                                                        'lipsLowerInner',\
                                                                        'rightEyeUpper0',\
                                                                        'rightEyeLower0',\
                                                                        #'rightEyeUpper1',\
                                                                        #'rightEyeLower1',\
                                                                        #'rightEyeUpper2',\
                                                                        #'rightEyeLower2',\
                                                                        #'rightEyeLower3',\
                                                                        #'rightEyebrowUpper',\
                                                                        'rightEyebrowLower',\
                                                                        'leftEyeUpper0',\
                                                                        'leftEyeLower0',\
                                                                        #'leftEyeUpper1',\
                                                                        #'leftEyeLower1',\
                                                                        #'leftEyeUpper2',\
                                                                        #'leftEyeLower2',\
                                                                        #'leftEyeLower3',\
                                                                        'leftEyebrowLower',\
                                                                        'midwayBetweenEyes',\
                                                                        #'noseBottom',\
                                                                        'noseRightCorner',\
                                                                        'noseLeftCorner',\
                                                                        'noseTip',\
                                                                        #'rightCheek',\
                                                                        #'leftCheek'                                                                                  
                                                                        ]
                                ]
    face_keypoints = sum(face_keypoints_temp, [])

    return hands_keypoints, pose_keypoints, face_keypoints

# define keypoints to keep
hands_keypoints, pose_keypoints, face_keypoints = define_keypoints_to_keep()


def get_hands_landmarks(frame, width, height, keypoints= hands_keypoints):
    '''
    Generate hands landmarks of the frame associated to selected keypoints.
    Paramaters
    ---------
    frame: numpy array, corresponding to the image
    width: int, width of the image
    height: int; height of the image
    keypoints: list of int, default value hands_keypoints return by funcion define_keypoints_to_keep

    Returns
    -------
    hands_lm_dict: dictionnary. For both hand return dictionnary with key the id of the keypoints()
    and value the x, y values of the landmarks associated.
    '''
    #calculate landmarks for the frame
    results_hands = hands.process(frame)
    left_hand_lm= []
    right_hand_lm = []
    # check if landmarks are detected in the frame
    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            # check if landmark are for left hand
            if results_hands.multi_handedness[i].classification[0].label == 'Left':
                #fill the left_hand_lm with approriated landmarks
                for keypoint, lm in enumerate(hand_landmarks.landmark):
                    if keypoint in keypoints:
                        left_hand_lm.append((keypoint, (int(width * lm.x), int(height *lm.y))))
            # repeat same process for right hand
            else:
                for keypoint, lm in enumerate(hand_landmarks.landmark):
                    if keypoint in keypoints:
                        right_hand_lm.append((keypoint, (int(width * lm.x), int(height *lm.y))))
                    
    # convert list to dictionnary
    left_hand_lm_dict = dict(left_hand_lm)
    right_hand_lm_dict = dict(right_hand_lm)
    hands_lm_dict = {'left_hand': left_hand_lm_dict, 'right_hand': right_hand_lm_dict}

    return hands_lm_dict


def get_pose_landmarks(frame, width, height,  keypoints=pose_keypoints):
    '''
    Generate pose landmarks of the frame associated to selected keypoints.
    Paramaters
    ---------
    frame: numpy array, corresponding to the image
    width: int, width of the image
    height: int; height of the image
    keypoints: list of int, default value pose_keypoints return by funcion define_keypoints_to_keep()

    Returns
    -------
    pose_lm_dict: dictionnary with key the id of the keypoints
    and values the x, y values of the landmarks associated.
    '''
    #calculate landmarks for the frame
    results_pose = pose.process(frame)
    pose_lm = []
    # check if landmarks are detected in the frame
    if results_pose.pose_landmarks:
        for keypoint, lm in enumerate(results_pose.pose_landmarks.landmark):
            if keypoint in keypoints:
                pose_lm.append((keypoint, (int(width * lm.x), int(height *lm.y))))

    # convert list to dictionnary
    pose_lm_dict = dict(pose_lm)
          
    return pose_lm_dict


def get_face_landmarks(frame, width, height,  keypoints=face_keypoints):
    '''
    Generate face landmarks of the frame associated to selected keypoints.
    Paramaters
    ---------
    frame: numpy array, corresponding to the image
    width: int, width of the image
    height: int; height of the image
    keypoints: list of int, default value face_keypoints return by funcion define_keypoints_to_keep()

    Returns
    -------
    face_lm_dict: dictionnary with key the id of the keypoints
    and values the x, y values of the landmarks associated.
    '''
    #calculate landmarks for the frame 
    results_face = face_mesh.process(frame)
    face_lm = []
    # check if landmarks are detected in the frame
    if results_face.multi_face_landmarks:
        for keypoint, lm in enumerate(results_face.multi_face_landmarks[0].landmark):
            if keypoint in keypoints:
                face_lm.append((keypoint, (int(width * lm.x), int(height *lm.y))))
    
    # convert list to dictionnary
    face_lm_dict = dict(face_lm)
          
    return face_lm_dict

def get_video_landmarks(video_path, start_frame=1, end_frame=-1,\
                        hands_keypoints=hands_keypoints,\
                        pose_keypoints=pose_keypoints,\
                        face_keypoints=face_keypoints):
    '''
    Generate all the landmarks of the frame associated to selected keypoints.
    Paramaters
    ---------
    video_path: str, path to the video file
    start_frame: int, frame index number that we want to start to read video
    end_frame: int, frame index number that we want to stop to read video
    hands_keypoints: list of int, default value=face_keypoints return by funcion define_keypoints_to_keep()
    pose_keypoints: list of int, default value=face_keypoints return by funcion define_keypoints_to_keep()
    face_keypoints: list of int, default value=face_keypoints return by funcion define_keypoints_to_keep()

    Returns
    -------
    video_landmarks: list of dictionnary, each dictionnary contains subdctionnary of landmarks for the selected
    hands, pose and face keypoints for a specific frame
    '''
    
    # open video file and get size of the frame
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # deal with frame index
    # if the starting is 0
    if start_frame <= 1:
        start_frame = 1
        
    # if the video is precropped
    elif start_frame > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        start_frame = 1
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    # if the final frame was not given (-1)    
    if end_frame < 0: 
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = 1
    
    video_landmarks=[]

    while cap.isOpened() and frame_index <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index >= start_frame:
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_landmarks.append({'hands_landmarks': get_hands_landmarks(frame, width, height, hands_keypoints),\
                                    'pose_landmarks': get_pose_landmarks(frame, width, height, pose_keypoints),\
                                    'face_landmarks': get_face_landmarks(frame, width, height, face_keypoints)})

        frame_index += 1
    
    # add width and height information
    video_landmarks.append({'width': width, 'height': height})
    cap.release()
    hands.reset()
    pose.reset()
    face_mesh.reset()
    return video_landmarks


def save_keypoints_to_json():
    '''
    create a folder video_landmarks and fill in with .json files of landmarks for each video of the dataset filtered_WLASL
    and update dataset filtered_WLASL with new column video_landmarks_path
    
    Parameters
    ----------
    None

    Return
    ------
    .json file for all the dataset and up date dataset
    '''
    dataset_filepath = 'local_dataset'
    data = pd.read_csv( dataset_filepath, dtype={'video_id': str})

    # Create video_landmarks folder if not exist
    directory = 'video_landmarks'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create json_file of landmark for each video of the dataset
    for _ , row in data.iterrows():
        video_path = row['video_path']
        video_id = row['video_id']
        filepath = os.path.join('video_landmarks', video_id)
        # check if .json file with video_id already exist in video_landmarks folder
        if os.path.isfile(filepath):
            continue
        #create .json file
        video_lm = get_video_landmarks(video_path)
        with open(filepath, 'w') as f:
            json.dump(video_lm, f)

if __name__ == '__main__':
    save_keypoints_to_json()
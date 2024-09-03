import cv2
import pandas as pd
import random as rd
import mediapipe as mp
import numpy as np
import json
import os


# ---- Dictionnary of correspondance between keypoints ID and area of the face
#
face_keypoints_correspondance =\
    {"silhouette": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,\
                    136, 172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109],
    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
    "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
    "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
    "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
    "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
    "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],
    "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
    "rightEyebrowLower": [35, 124, 46, 53, 52, 65],
    "rightEyeIris": [473, 474, 475, 476, 477],
    "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
    "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
    "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
    "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
    "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
    "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],
    "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
    "leftEyebrowLower": [265, 353, 276, 283, 282, 295],
    "leftEyeIris": [468, 469, 470, 471, 472],
    "midwayBetweenEyes": [168],
    "noseTip": [1],
    "noseBottom": [2],
    "noseRightCorner": [98],
    "noseLeftCorner": [327],
    "rightCheek": [205],
    "leftCheek": [425]}

# ---- Instanciate mediapipe models for hands, posture and face keypoints detection
#
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)


def define_keypoints_to_keep(face_keypoints_correspondance=face_keypoints_correspondance):
    '''
    Select the specific keypoints to be retained from those returned by Mediapipe models.

    Parameters
    ----------
    face_keypoints_correspondance : dict
        A dictionary that maps parts of the face to their corresponding keypoint IDs.

    Returns
    -------
    tuple
        A tuple containing three lists:
        - hands_keypoints : list of int
            A list of keypoint IDs for the hands.
        - pose_keypoints : list of int
            A list of keypoint IDs for the pose.
        - face_keypoints : list of int
            A list of keypoint IDs for the face.
    '''

    # ---- Keep all hand keypoints (21 keypoints per hand)
    #
    hands_keypoints = list(range(21))

    # ---- Keep only selected pose keypoints (shoulders, elbows, wrists)
    #
    pose_keypoints = [11, 12, 13, 14, 15, 16]

    # ---- Keep selected face keypoints based on the provided correspondence
    #
    selected_face_areas = [
        'lipsLowerInner', 'rightEyeUpper0', 'rightEyeLower0', 'rightEyebrowLower',
        'leftEyeUpper0', 'leftEyeLower0', 'leftEyebrowLower', 'midwayBetweenEyes',
        'noseRightCorner', 'noseLeftCorner', 'noseTip'
    ]

    face_keypoints_temp = [face_keypoints_correspondance[keypoint_area] for keypoint_area in selected_face_areas]
    
    face_keypoints = sum(face_keypoints_temp, [])

    return hands_keypoints, pose_keypoints, face_keypoints


def get_hands_landmarks(frame, width, height, keypoints):
    '''
    Generate hand landmarks for the given frame, focusing on selected keypoints for both hands.

    Parameters
    ----------
    frame : numpy.ndarray
        The image frame in which hand landmarks are to be detected.
    width : int
        The width of the frame.
    height : int
        The height of the frame.
    keypoints : list of int
        A list of hand keypoint IDs to be selected.

    Returns
    -------
    hands_lm_dict : dict
        A dictionary containing two dictionaries, one for each hand ('left_hand' and 'right_hand').
        Each of these dictionaries maps hand keypoint IDs to tuples of (x, y) coordinates.
    '''

    # ---- Calculate landmarks for the frame
    #
    results_hands = hands.process(frame)
    left_hand_lm = []
    right_hand_lm = []

    # ---- Check if landmarks are detected in the frame
    #
    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            # ---- Check if landmark are for left hand
            # 
            if results_hands.multi_handedness[i].classification[0].label == 'Left':
                # ---- Populate left_hand_lm with tuples of keypoint ID and corresponding (x, y) coordinates
                #
                for keypoint, lm in enumerate(hand_landmarks.landmark):
                    if keypoint in keypoints:
                        left_hand_lm.append((keypoint, (int(width * lm.x), int(height *lm.y))))
            # ---- Repeat same process for right hand
            #
            else:
                for keypoint, lm in enumerate(hand_landmarks.landmark):
                    if keypoint in keypoints:
                        right_hand_lm.append((keypoint, (int(width * lm.x), int(height *lm.y))))

    # ---- Convert list to dictionnary
    #
    left_hand_lm_dict = dict(left_hand_lm)
    right_hand_lm_dict = dict(right_hand_lm)
    hands_lm_dict = {'left_hand': left_hand_lm_dict, 'right_hand': right_hand_lm_dict}

    return hands_lm_dict


def get_pose_landmarks(frame, width, height, keypoints):
    '''
    Generate posture landmarks for the given frame, focusing on selected keypoints for both hands.

    Parameters
    ----------
    frame : numpy.ndarray
        The image frame in which posture landmarks are to be detected.
    width : int
        The width of the frame.
    height : int
        The height of the frame.
    keypoints : list of int
        A list of posture keypoint IDs to be selected.

    Returns
    -------
    pose_lm_dict : dict
        A dictionary containing mapping posture keypoint IDs to tuples of (x, y) coordinates.
    '''

    # ---- Calculate landmarks for the frame
    #
    results_pose = pose.process(frame)
    pose_lm = []

    # ---- Check if landmarks are detected in the frame
    #
    if results_pose.pose_landmarks:
        for keypoint, lm in enumerate(results_pose.pose_landmarks.landmark):
            if keypoint in keypoints:
                pose_lm.append((keypoint, (int(width * lm.x), int(height *lm.y))))

    # ---- Convert list to dictionnary
    #
    pose_lm_dict = dict(pose_lm)

    return pose_lm_dict


def get_face_landmarks(frame, width, height, keypoints):
    '''
    Generate face landmarks for the given frame, focusing on selected keypoints for both hands.

    Parameters
    ----------
    frame : numpy.ndarray
        The image frame in which face landmarks are to be detected.
    width : int
        The width of the frame.
    height : int
        The height of the frame.
    keypoints : list of int
        A list of face keypoint IDs to be selected.

    Returns
    -------
    face_lm_dict : dict
        A dictionary containing mapping face keypoint IDs to tuples of (x, y) coordinates.
    '''

    # ---- Calculate landmarks for the frame
    # 
    results_face = face_mesh.process(frame)
    face_lm = []

    # ---- Calculate landmarks for the frame
    #
    if results_face.multi_face_landmarks:
        for keypoint, lm in enumerate(results_face.multi_face_landmarks[0].landmark):
            if keypoint in keypoints:
                face_lm.append((keypoint, (int(width * lm.x), int(height *lm.y))))

    # ---- Convert list to dictionnary
    #
    face_lm_dict = dict(face_lm)

    return face_lm_dict

def get_video_landmarks(video_path,\
                        hands_keypoints,\
                        pose_keypoints,\
                        face_keypoints):
    '''
    Generate landmarks for selected keypoints from each frame of the given video.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    hands_keypoints : list of int
        IDs of the hand keypoints to be extracted.
    pose_keypoints : list of int
        IDs of the pose keypoints to be extracted.
    face_keypoints : list of int
        IDs of the face keypoints to be extracted.

    Returns
    -------
    video_landmarks : list of dict
        A list where each dictionary contains the landmarks for the selected hand, pose, and face keypoints for a specific frame.
        The last dictionary in the list contains the width and height of the video.
    '''

    # ---- Open the video file and get frame characteristics
    #
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- Initialize values
    #
    frame_index = 1
    video_landmarks=[]

    # ---- Loop over each frame
    #
    while cap.isOpened() and frame_index <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- Populate video_landmarks list with a dictionnary of keypoints and landmarks of the current frame
        #
        video_landmarks.append({'hands_landmarks': get_hands_landmarks(frame, width, height, hands_keypoints),\
                                    'pose_landmarks': get_pose_landmarks(frame, width, height, pose_keypoints),\
                                    'face_landmarks': get_face_landmarks(frame, width, height, face_keypoints)})

        frame_index += 1

    # ---- Add video width and height information to the list
    #
    video_landmarks.append({'width': width, 'height': height})
    cap.release()

    # ---- Reset the models used for landmark detection
    #
    cap.release()
    hands.reset()
    pose.reset()
    face_mesh.reset()

    return video_landmarks


def save_keypoints_to_json(hands_keypoints, pose_keypoints, face_keypoints, dataset='enhanced_dataset'):
    '''
    Create a folder named 'video_landmarks' and populate it with JSON files containing keypoints and landmarks 
    for each video in the dataset. The dataset is then updated with a new column 'video_landmarks_path' indicating 
    the path to the corresponding JSON file for each video.

    Parameters
    ----------
    
    hands : mp.solutions.hands.Hands
        The Mediapipe Hands model instance.
    pose : mp.solutions.pose.Pose
        The Mediapipe Pose model instance.
    face_mesh : mp.solutions.face_mesh.FaceMesh
        The Mediapipe Face Mesh model instance.
    hands_keypoints : list of int
        IDs of the hand keypoints to be extracted.
    pose_keypoints : list of int
        IDs of the pose keypoints to be extracted.
    face_keypoints : list of int
        IDs of the face keypoints to be extracted.
    dataset : str
        Path to the dataset CSV file containing information about the videos used to generate keypoints.

    Returns
    -------
    None
        Creates a new folder named 'video_landmarks' if it doesn't already exist.
        Saves JSON files of keypoints and landmarks for each video in the dataset.
        Updates the dataset with a new column 'video_landmarks_path' and saves it.
    '''

    data = pd.read_csv(dataset)

    # ---- Create 'video_landmarks' folder if not exist
    #
    directory = 'video_landmarks'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # ---- Loop over each row of the dataset
    #
    for _ , row in data.iterrows():
        # ---- Get video local path and video_id
        #
        video_path = os.path.join('..', row['video_path'])
        video_id = str(row['video_id'])

        # ---- Generate the list of dictionnary containing keypoints and landmarks for the current video
        #
        video_lm = get_video_landmarks(video_path, hands_keypoints=hands_keypoints,\
                        pose_keypoints=pose_keypoints,\
                        face_keypoints=face_keypoints)

        # ---- Save all landmarks for the video in a JSON file
        #
        filepath = os.path.join('video_landmarks', video_id)
        with open(filepath, 'w') as f:
            json.dump(video_lm, f)

        # ---- Update dataset with the new column video_landmarks_path
        #
        data['video_landmarks_path'] = filepath

    # ---- Overwrite the dataset with the updated data
    #
    data.to_csv(dataset, index=False)


def main():
    
    # ---- define keypoints to keep
    #
    hands_keypoints, pose_keypoints, face_keypoints = define_keypoints_to_keep()

    # ---- Save keypoints in JSON file in a specific folder
    #
    save_keypoints_to_json(hands_keypoints, pose_keypoints, face_keypoints)


if __name__ == "__main__":
    main()

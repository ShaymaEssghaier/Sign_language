import cv2
import json
import numpy as np
import pandas as pd
import time


def draw_hands_connections(frame, hand_landmarks):
    '''
    Draw white lines on the given frame between relevant hand keypoints.

    Parameters
    ----------
    frame: numpy array
        The frame on which we want to draw.
    hand_landmarks: dict
        Dictionary mapping keypoint IDs (integers) to hand landmarks 
        (lists of two floats corresponding to the coordinates) for both hands.

    Returns
    -------
    frame: numpy array
        The frame with the newly drawn hand connections.
    '''

    # ---- Define hand_connections between keypoints to draw
    #
    hand_connections = [[0, 1], [1, 2], [2, 3], [3, 4],
                        [5, 6], [6, 7], [7, 8],
                        [9, 10], [10, 11], [11, 12],
                        [13, 14], [14, 15], [15, 16],
                        [17, 18], [18, 19], [19, 20]] #[5, 2], [0, 17]]
    
    # ---- loop to draw left hand connections
    #
    for connection in hand_connections:
        landmark_start = hand_landmarks['left_hand'].get(str(connection[0]))
        landmark_end = hand_landmarks['left_hand'].get(str(connection[1]))
        cv2.line(frame, landmark_start, landmark_end, (255, 255, 255), 2)
    
    # ---- loop to to draw right hand connections
    #
    for connection in hand_connections:
        landmark_start = hand_landmarks['right_hand'].get(str(connection[0]))
        landmark_end = hand_landmarks['right_hand'].get(str(connection[1]))
        cv2.line(frame, landmark_start, landmark_end, (255, 255, 255), 2)
    
    return frame

def draw_pose_connections(frame, pose_landmarks):
    '''
    Draw white lines on the given frame between relevant posture keypoints.

    Parameters
    ----------
    frame: numpy array
        The frame on which we want to draw.
    pose_landmarks: dict
        Dictionary mapping keypoint IDs (integers) to posture landmarks 
        (lists of two floats corresponding to the coordinates).

    Returns
    -------
    frame: numpy array
        The frame with the newly drawn posture connections.
    '''

    # ---- define posture connections between keypoints to draw
    #
    pose_connections = [[11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]

    # ---- loop to to draw posture connections
    #
    for connection in pose_connections:
        landmark_start = pose_landmarks.get(str(connection[0]))
        landmark_end = pose_landmarks.get(str(connection[1]))
        cv2.line(frame, landmark_start, landmark_end, (255, 255, 255), 2)

    return frame

def draw_face_connections(frame, face_landmarks):
    '''
    Draw white lines on the given frame between relevant face keypoints.

    Parameters
    ----------
    frame: numpy array
        The frame on which we want to draw.
    face_landmarks: dict
        Dictionary mapping keypoint IDs (integers) to face landmarks 
        (lists of two floats corresponding to the coordinates).

    Returns
    -------
    frame: numpy array
        The frame with the newly drawn face connections.
    '''
    # ---- define pose connections
    #
    connections_dict = {'lipsUpperInner_connections' : [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],\
                    'lipsLowerInner_connections' : [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],\
                    'rightEyeUpper0_connections': [246, 161, 160, 159, 158, 157, 173],\
                    'rightEyeLower0' : [33, 7, 163, 144, 145, 153, 154, 155, 133],\
                    'rightEyebrowLower' : [35, 124, 46, 53, 52, 65],\
                    'leftEyeUpper0' : [466, 388, 387, 386, 385, 384, 398],\
                    'leftEyeLower0' : [263, 249, 390, 373, 374, 380, 381, 382, 362],\
                    'leftEyebrowLower' : [265, 353, 276, 283, 282, 295],\
                    'noseTip_midwayBetweenEye' :  [1, 168],\
                    'noseTip_noseRightCorner' : [1, 98],\
                    'noseTip_LeftCorner' : [1, 327]\
                    }

    # ---- loop to to draw face connections
    #
    for keypoints_list in connections_dict.values():
        for index in range(len(keypoints_list)):
            if index + 1 < len(keypoints_list):
                landmark_start = face_landmarks.get(str(keypoints_list[index]))
                landmark_end = face_landmarks.get(str(keypoints_list[index+1]))
                cv2.line(frame, landmark_start, landmark_end, (255, 255, 255), 1)
    return frame

def resize_landmarks(landmarks, resize_rate_width, resize_rate_height):
    '''
    Resize landmark coordinates by applying specific scaling factors 
    to both the width and height of the frame.

    Parameters
    ----------
    landmarks: dict
        Dictionary mapping keypoint IDs (integers) to landmarks
        (lists of two floats corresponding to the coordinates).
    resize_rate_width: float
        Scaling factor applied to the x-coordinate (width).
    resize_rate_height: float
        Scaling factor applied to the y-coordinate (height).

    Returns
    -------
    landmarks: dict
        Dictionary mapping keypoint IDs (integers) to the newly resized landmarks
        (lists of two integers corresponding to the coordinates).
    '''

    for keypoint in landmarks.keys():
        landmark_x, landmark_y = landmarks[keypoint]
        landmarks[keypoint] = [int(resize_rate_width * landmark_x), int(resize_rate_height*landmark_y)]

    return landmarks

def generate_video(gloss_list, dataset, vocabulary_list):
    '''
    Generate a video stream from a list of glosses.

    Parameters
    ----------
    gloss_list: list of str
        List of glosses from which the signing video will be generated.
    dataset: pandas.DataFrame
        Dataset containing information about each gloss, including paths to landmark data.
    vocabulary_list: list of str
        List of tokens that have associated landmarks collected.

    Yields
    ------
    frame: bytes
        JPEG-encoded frame for streaming.
    '''
    # ---- Fix size of the frame to the most common size of video we have in the dataset
    # (corresponding to signer ID 11 who has the maximum number of videos).
    #
    FIXED_WIDTH,  FIXED_HEIGHT = 576, 384

    # ---- Fix the Frames Per Second (FPS) to match the videos collected in the dataset.
    #
    FPS = 25

    # ---- Define carachteristics for text display.
    #
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)
    thickness = 2
    line_type = cv2.LINE_AA

    # ---- Loop over each gloss
    #
    for gloss in gloss_list:
        # ---- Skip if gloss not in the vocabulary_list.
        #
        if not check_gloss_in_vocabulary(gloss, vocabulary_list):
            continue

        # ---- Get landmarks of all the frame in the dataset corresponding to the appropriate gloss.
        #
        video_id = select_video_id_from_gloss(gloss, dataset)
        video_landmarks_path = dataset.loc[dataset['video_id'] == video_id, 'video_landmarks_path'].values[0]
        with open(video_landmarks_path, 'r') as f:
            video_landmarks = json.load(f)
        width = video_landmarks[-1].get('width')
        height = video_landmarks[-1].get('height')

        # ---- Calculate resize rate for future landmark rescaling.
        #
        resize_rate_width, resize_rate_height  = FIXED_WIDTH / width, FIXED_HEIGHT/height

        # ---- Loop over each frame
        #
        for frame_landmarks in video_landmarks[:-1]:
            # ---- Initialize blank image and get all landmarks of the given frame.
            #
            blank_image = np.zeros((FIXED_HEIGHT, FIXED_WIDTH, 3), dtype=np.uint8)
            frame_hands_landmarks = frame_landmarks['hands_landmarks']
            frame_pose_landmarks = frame_landmarks['pose_landmarks']
            frame_face_landmarks = frame_landmarks['face_landmarks']

            # ---- Resize landmarks.
            #
            frame_hands_landmarks_rs = {
                            'left_hand': resize_landmarks(frame_hands_landmarks['left_hand'], resize_rate_width, resize_rate_height),
                            'right_hand': resize_landmarks(frame_hands_landmarks['right_hand'], resize_rate_width, resize_rate_height)
                                        }
            frame_pose_landmarks_rs = resize_landmarks(frame_pose_landmarks, resize_rate_width, resize_rate_height)
            frame_face_landmarks_rs = resize_landmarks(frame_face_landmarks, resize_rate_width, resize_rate_height)
            
            # ---- Draw relevant connections between keypoints on the frame.
            #
            draw_hands_connections(blank_image, frame_hands_landmarks_rs)
            draw_pose_connections(blank_image, frame_pose_landmarks_rs)
            draw_face_connections(blank_image, frame_face_landmarks_rs)

            # ---- Display text corresponding to the gloss on the frame.
            #
            text_size, _ = cv2.getTextSize(gloss, font, font_scale, thickness)
            text_x = (FIXED_WIDTH - text_size[0]) // 2
            text_y = FIXED_HEIGHT - 10
            cv2.putText(blank_image, gloss, (text_x, text_y), font, font_scale, font_color, thickness, line_type)
            
             # ---- JPEG-encode the frame for streaming.
            #
            _, buffer = cv2.imencode('.jpg', blank_image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            time.sleep(1 / FPS)


def load_data(dataset_path='enhanced_dataset'):
    '''
    Load the dataset that contains all information about glosses.

    Parameters
    ----------
    dataset_path: str
        Local path to the dataset.

    Returns
    -------
    data_df: pandas.DataFrame
        DataFrame containing the dataset with information about each gloss.
    vocabulary_list: list of str
        List of glosses (tokens) that have associated landmarks collected.
    '''

    filepath = dataset_path
    data_df = pd.read_csv(filepath, dtype={'video_id': str})
    vocabulary_list = data_df['gloss'].tolist()

    return data_df, vocabulary_list


def check_gloss_in_vocabulary(gloss, vocabulary_list):
    '''
    Check if the given gloss is in the vocabulary list.

    Parameters
    ----------
    gloss: str
        The gloss to check.
    vocabulary_list: list of str
        List of glosses (tokens) that have associated landmarks collected.

    Returns
    -------
    bool
        True if the gloss is in the vocabulary list, False otherwise.
    '''

    return gloss in vocabulary_list


def select_video_id_from_gloss(gloss, dataset):
    '''
    Selects a video ID corresponding to the given gloss from the dataset.

    Parameters
    ----------
    gloss : str
        The gloss for which to retrieve the video ID.
    dataset : pandas.DataFrame
        A DataFrame containing information about each gloss, including 'signer_id', 'gloss', and 'video_id'.

    Returns
    -------
    int
        The video ID corresponding to the given gloss. If the gloss is found for 'signer_id' 11, the video ID for that signer is returned; otherwise, the video ID for the gloss from the entire dataset is returned.
    '''
    # ---- Choose preferentialy ID 11 because this signer with this ID signed the more video
    #
    filtered_data_id_11 = dataset.loc[dataset['signer_id'] == 11]

    if gloss in filtered_data_id_11['gloss'].tolist():
        video_id = filtered_data_id_11.loc[filtered_data_id_11['gloss'] == gloss, 'video_id'].values
    else:
        video_id = dataset.loc[dataset['gloss'] == gloss, 'video_id'].values

    return video_id[0]
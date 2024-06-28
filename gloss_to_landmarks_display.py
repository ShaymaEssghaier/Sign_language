import pandas as pd
import json
import numpy as np
import cv2
import os

def draw_hands_connections(frame, hand_landmarks):
    '''
    Draw white lines between relevant points of hands landmarks
    
    Parameters
    ----------
    frame: numpy array, corresponding to the frame on which we want to draw
    hand_landmarks: dictionnary, collecting the hands landmarks

    Return
    ------
    frame: numpy array, with the newly drawing of the hands
    '''
    # define hand_connections between keypoints
    hand_connections = [[0, 1], [1, 2], [2, 3], [3, 4],
                        [5, 6], [6, 7], [7, 8],
                        [9, 10], [10, 11], [11, 12],
                        [13, 14], [14, 15], [15, 16],
                        [17, 18], [18, 19], [19, 20]]
    
    # loop to draw left hand connection
    for connection in hand_connections:
        landmark_start = hand_landmarks['left_hand'].get(str(connection[0]))
        landmark_end = hand_landmarks['left_hand'].get(str(connection[1]))
        cv2.line(frame, landmark_start, landmark_end, (255, 255, 255), 2)
    
    # loop to to draw right hand connection
    for connection in hand_connections:
        landmark_start = hand_landmarks['right_hand'].get(str(connection[0]))
        landmark_end = hand_landmarks['right_hand'].get(str(connection[1]))
        cv2.line(frame, landmark_start, landmark_end, (255, 255, 255), 2)
    
    return frame

def draw_pose_connections(frame, pose_landmarks):
    '''
    Draw white lines between relevant points of pose landmarks
    
    Parameters
    ----------
    frame: numpy array, corresponding to the frame on which we want to draw
    hand_landmarks: dictionnary, collecting the pose landmarks
    
    Return
    ------
    frame: numpy array, with the newly drawing of the pose
    '''
    # define pose connections
    pose_connections = [[11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]

    for connection in pose_connections:
        landmark_start = pose_landmarks.get(str(connection[0]))
        landmark_end = pose_landmarks.get(str(connection[1]))
        cv2.line(frame, landmark_start, landmark_end, (255, 255, 255), 2)

    return frame

def draw_face_connections(frame, face_landmarks):
    '''
    Draw white lines between relevant points of face landmarks
    
    Parameters
    ----------
    frame: numpy array, corresponding to the frame on which we want to draw
    hand_landmarks: dictionnary, collecting the face landmarks
    
    Return
    ------
    frame: numpy array, with the newly drawing of the face
    '''
    # define pose connections
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

    for keypoints_list in connections_dict.values():
        for index in range(len(keypoints_list)):
            if index + 1 < len(keypoints_list):
                landmark_start = face_landmarks.get(str(keypoints_list[index]))
                landmark_end = face_landmarks.get(str(keypoints_list[index+1]))
                cv2.line(frame, landmark_start, landmark_end, (255, 255, 255), 2)
    return frame

def display_landmarks(gloss, video_id, dataset):
    '''
    Generate a video from the landmarks of the video with black background
    
    Parameters
    ----------
    gloss: str, the name of the gloss to display
    video_id: int, the id of the initial video from which landmarks have been build
    FPS: int, the number of images per sec of the initial video from which landmarks have been build
    
    Return
    ------
    None: display video
    '''
    # get the frame per second of the initial video
    fps = dataset.loc[dataset['video_id'] == video_id, 'fps'].values[0]

    # load landmarks for the video
    video_landmarks_path = 'data/' + dataset.loc[dataset['video_id'] == video_id, 'video_landmarks_path'].values[0]
    with open(video_landmarks_path, 'r') as f:
        video_landmarks = json.load(f)
    
    # get original size
    width = video_landmarks[-1].get('width')
    height = video_landmarks[-1].get('height')
    delay = int(1000 / fps)

    # Text to display
    text = gloss
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0) 
    thickness = 2
    line_type = cv2.LINE_AA

     # Create the window to display
    window_name = 'Hands, Pose and FaceMesh Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # Fix a fixed dimension for the frame and the window
    FIXED_HEIGHT, FIXED_WIDTH =  480 , 640
     # define location for window and resize
    cv2.moveWindow(window_name, 100, 100)  # location (100, 100) on the screen
    cv2.resizeWindow(window_name, FIXED_WIDTH, FIXED_HEIGHT)

    for frame_landmarks in video_landmarks[:-1]:

        # initialize black background
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)

        # dispatch landmarks
        frame_hands_landmarks = frame_landmarks['hands_landmarks']
        frame_pose_landmarks = frame_landmarks['pose_landmarks']
        frame_face_landmarks = frame_landmarks['face_landmarks']

        # draw hands_landmarks
        left_hand_landmarks_xy = [(x, y) for x, y in frame_hands_landmarks['left_hand'].values()]
        right_hand_landmarks_xy = [(x, y) for x, y in frame_hands_landmarks['right_hand'].values()]
        for x, y in left_hand_landmarks_xy:
            cv2.circle(blank_image, (x, y), 2, (255, 0, 0), -1)
        for x, y in right_hand_landmarks_xy:
            cv2.circle(blank_image, (x, y), 2, (255, 0, 0), -1)

        # draw pose_landmarks
        pose_landmarks_xy = [(x, y) for x, y in frame_pose_landmarks.values()]
        for x, y in  pose_landmarks_xy:
            cv2.circle(blank_image, (x, y), 2, (255, 0, 0), -1)

        # draw face_landmarks
        face_landmarks_xy = [(x, y) for x, y in frame_face_landmarks.values()]
        for x, y in  face_landmarks_xy:
            cv2.circle(blank_image, (x, y), 2, (255, 0, 0), -1)

        # draw connections between landmarks
        draw_hands_connections(blank_image, frame_hands_landmarks)
        draw_pose_connections(blank_image, frame_pose_landmarks)
        draw_face_connections(blank_image, frame_face_landmarks)

        # defnie location for text
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (FIXED_WIDTH - text_size[0]) // 2
        text_y = FIXED_HEIGHT - 10

        # Resize frame to get fixed size whatever the original video
        blank_image = cv2.resize(blank_image, (FIXED_WIDTH, FIXED_HEIGHT))

        # Add text to frame
        cv2.putText(blank_image, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)
    
        cv2.imshow(window_name, blank_image)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # destroy window OpenCV
    #cv2.destroyAllWindows()

def load_data(dataset_path='data/local_dataset'):
    ''' 
    load video_dataset
    
    parameter:
    ---------
    dataset: str, name of the csv dataset

    returns:
    --------
    data: pd.DataFrame
    vocabulary_list: list of str
    '''
    filepath = os.path.join(dataset_path)
    data_df = pd.read_csv(filepath, dtype={'video_id': str})
    vocabulary_list = data_df['gloss'].tolist()

    return data_df, vocabulary_list

def check_gloss_in_vocabulary(gloss, vocabulary_list):
    '''
    check if the gloss is in the dataset

    parameter:
    ---------
    gloss: str

    returns:
    --------
    boolean
    '''
    
    if gloss in vocabulary_list:
        return True
    else:
        return False
    

def select_video_id_from_gloss(gloss, dataset, vocabulary_list):
    '''
    extract the landmarks_path from the video matching with the gloss. Preferentialy for signer with signer_id 11 that made more video

    parameter:
    ---------
    gloss: str

    returns:
    --------
    filepath: str
    '''
    #signer with signer_id=11 is the signer with the maximum video in dataset filtered_WASL.csv
    filtered_data_id_11 = dataset.loc[dataset['signer_id'] == 11]

    if check_gloss_in_vocabulary(gloss, vocabulary_list):
        # select preferentialy a video from signer_id 11

        if gloss in filtered_data_id_11['gloss'].tolist():
            video_id = filtered_data_id_11.loc[filtered_data_id_11['gloss'] == gloss, 'video_id'].values
        else:
            video_id = dataset.loc[dataset['gloss'] == gloss, 'video_id'].values
            
        return video_id[0] #index 0 in case of the signer did several video for the same gloss


def display_serie_gloss (gloss_list):
    '''
    display a continuity a serie of video, each video matching with a gloss

    Parameters
    ----------
    gloss_list: list of str

    Returns
    -------
    None
    '''
    dataset, vocabulary_list = load_data()

    for gloss in gloss_list:
        if check_gloss_in_vocabulary(gloss, vocabulary_list) == False:
            print(gloss)
            continue
        video_id = select_video_id_from_gloss(gloss, dataset, vocabulary_list)
        display_landmarks(gloss, video_id, dataset)
    
    cv2.destroyAllWindows()


gloss_list=['a', 'a lot', 'abdomen', 'accent', 'active']

display_serie_gloss (gloss_list)


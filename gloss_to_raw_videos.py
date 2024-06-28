import pandas as pd
import os
import cv2

def get_video_dataset(dataset='data/local_dataset'):
    ''' 
    load video_dataset
    
    parameter:
    ---------
    dataset: name of the csv dataset

    returns:
    --------
    video_df: DataFrame
    vocabulary_list: list of str
    '''

    filepath = os.path.join(dataset)
    video_df = pd.read_csv(filepath, dtype={'video_id': str})
    vocabulary_list = video_df['gloss'].tolist()

    return video_df, vocabulary_list


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
    
def get_video_path_from_gloss(gloss, dataset, vocabulary_list):

    '''
    extract the video_path from the video matching with the gloss. Preferentialy for signer with signer_id 11 that made more video

    parameter:
    ---------
    gloss: str

    returns:
    --------
    filepath: str
    '''
    filtered_df_id_11 = dataset.loc[video_df['signer_id'] == 11] #signer_id with the maximum video in dataset real_dat.csv

    if check_gloss_in_vocabulary(gloss, vocabulary_list):
        # select preferentialy a video from signer_id 11
        if gloss in filtered_df_id_11['gloss'].tolist():
            # display video
            video_path = filtered_df_id_11.loc[filtered_df_id_11['gloss'] == gloss, 'video_path'].values
        else:
            video_path = dataset.loc[dataset['gloss'] == gloss, 'video_path'].values
    
        current_directory = os.getcwd()
        
        return os.path.join(current_directory, video_path[0]) #index 0 in case of the signer did several video for the same gloss

def display_video_from_gloss(gloss, dataset, vocabulary_list):
    '''
    display the video matching with the gloss
    '''

    video_path = get_video_path_from_gloss(gloss, dataset, vocabulary_list)

    # open video from video_path
    video_capture = cv2.VideoCapture(video_path)
    print(video_path)

    if not video_capture.isOpened():
        print("Error with the opening of the video")

    # Text to display
    text = gloss
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0) 
    thickness = 2
    line_type = cv2.LINE_AA

     # Create the window to display
    window_name = 'Frame'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Get the frames per second (fps) of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # Fail-safe in case fps is not retrieved correctly
        fps = 25  # Default to 30 fps
    delay = int(1000 / fps)  # Convert fps to milliseconds per frame

    # Loop to read each frame of the video
    while video_capture.isOpened():
        ret, frame = video_capture.read()  # read the next frame

        if not ret:
            break  # Quit if video is finished

        # Get dimension of the frame
        #height, width, _ = frame.shape
        height, width =  480 , 640

        # Déplacer et redimensionner la fenêtre pour qu'elle corresponde à la taille de la vidéo
        cv2.moveWindow(window_name, 100, 100)  # Positionner la fenêtre à (100, 100) sur l'écran
        cv2.resizeWindow(window_name, width, height)

        # Calculate location for text
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (width - text_size[0]) // 2
        text_y = height - 10

         # Resize frame to get fixed size whatever the original video
        frame = cv2.resize(frame, (width, height))

        # Add text to frame
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Attendre 25ms et quitter si la touche 'q' est pressée
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Release capture
    video_capture.release()
    #cv2.destroyAllWindows()

def display_serie_gloss (gloss_list, dataset, vocabulary_list):

    '''
    display a continuity a serie of video, each video matching with a gloss
    '''

    for gloss in gloss_list:
        if check_gloss_in_vocabulary(gloss, vocabulary_list) == False:
            print(gloss)
            continue

        display_video_from_gloss(gloss, dataset, vocabulary_list)
    
    cv2.destroyAllWindows()

#test funcion

dataset, vocabulary_list = get_video_dataset()
gloss_list_test=['gym', 'cry', 'football', 'have', 'fun']

display_serie_gloss (gloss_list_test, dataset, vocabulary_list)
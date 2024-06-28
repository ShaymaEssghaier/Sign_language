import pandas as pd
import os

def update_video_path():
    '''
    Update or create local_dataset.csv by adding 'video_path' column fo filtered_WLASL_1999gloss
    '''
    data = pd.read_csv('src/filtered_WLASL_1999gloss', index_col=0, dtype={'video_id': str})

    # check if raw_videos folder exist
    if os.path.exists('raw_videos'):
        for file in os.listdir('raw_videos'):
            video_id = file.split('.')[0]
            data.loc[data['video_id'] == video_id, 'video_path'] = os.path.join('raw_videos', file)
        data.reset_index(drop=True)
        data.to_csv('local_dataset')

def update_video_landmarks_path():
    '''
    Update local_dataset.csv by adding 'video_landmarks_path' column
    '''
    data = pd.read_csv('local_dataset', index_col=0, dtype={'video_id': str})

    # check if video_landmarks folder exist
    if os.path.exists('video_landmarks'):
        for file in os.listdir('video_landmarks'):
            data.loc[data['video_id'] == file, 'video_landmarks_path'] = os.path.join('video_landmarks', file)
        data.reset_index(drop=True)
        data.to_csv('local_dataset', index=False)

if __name__ == '__main__':
    update_video_path()
    update_video_landmarks_path()
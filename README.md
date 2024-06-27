# Sign_language

Datasets links: 
text to gloss translation : https://huggingface.co/datasets/achrafothman/aslg_pc12
gloss to video : we use the file filtered_WLASL.csv. You can find more informatin about the data in the file data/data_description.rm

The downloading process of the video is inspired from the github repositery https://github.com/dxli94/WLASL where the data come from.
In the filtered dataset we build we didn't select youtube video, but here is the complete downloading process including for youtube video in case of we decide to add youtube videos from WLASL_v0.3.json dataset to filtered_WLASL.csv
## Download Videos
# Download repo.
git clone repo.link
Install youtube-dl for downloading YouTube videos.(not useful if you dont add youtube video to filtered_WLASL.csv)
# Download raw videos.
cd data
python video_downloader.py

# Generate landmarks from videos
cd data
pyton get_keypoints.py
(this command will:
- create a new folder in video_landmarks filled with video_id.json files for each videos,
- update filtered_WLASL.csv dataset by adding a new column 'video_landmarks_path')
# Sign_language

Datasets links: 
text to gloss translation : https://huggingface.co/datasets/achrafothman/aslg_pc12
gloss to video : we use the file filtered_WLASL_1999gloss.csv. You can find more informatin about the data in the file data/data_description.rm

The downloading process of the videos is inspired from the github repositery https://github.com/dxli94/WLASL where the data come from.
In the filtered dataset we build we didn't select youtube video, but here is the complete downloading process including for youtube video in case of we decide to add youtube videos from WLASL_v0.3.json dataset to filtered_WLASL_1999gloss.csv
## Download Videos
# Download repo.
git clone repo.link
Install youtube-dl for downloading YouTube videos.(not useful if you dont add youtube video to filtered_WLASL.csv)
# Download raw videos.
cd data
python video_downloader.py

(this command will download locally the videos with url link in filtered_WLASL_1999gloss.csv, in the folder data/raw_videos/)

python update_local_dataset.py

(this command will create a new .csv file called 'local_dataset.csv' in data/. It's a version of filtered_WLASL_1999gloss.csv with a new column 'video_path' for the locally path of the video downloaded).

# Generate landmarks from videos
cd data
pyton get_keypoints.py 

(this command will create a new folder in data/video_landmarks filled with video_id.json files for each videos)

python update_local_dataset.py

(this command will update 'local_dataset.csv' dataset by adding a new column 'video_landmarks_path' for the video with landmarks calculated)

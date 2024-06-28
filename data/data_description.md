# Sign_language

Datasets links: 
gloss to video : we use the file filtered_WLASL_1999gloss.csv that we build from the WLASL_v0.3.json avalaible on the github repositery https://github.com/dxli94/WLASL.

we adapt the downloading process of the videos from this github repositery https://github.com/dxli94/WLASL, also.
In the filtered dataset we build we didn't select youtube video, but here is the complete downloading process including for youtube video in case of we decide to add youtube videos from WLASL_v0.3.json dataset to filtered_WLASL_1999gloss.csv

Data Description
gloss: str, data file is structured/categorised based on sign gloss, or namely, labels.

bbox: [int], bounding box detected using YOLOv3 of (xmin, ymin, xmax, ymax) convention. Following OpenCV convention, (0, 0) is the up-left corner.

fps: int, frame rate (=25) used to decode the video as in the paper.

frame_start: int, the starting frame of the gloss in the video (decoding with FPS=25), indexed from 1.

frame_end: int, the ending frame of the gloss in the video (decoding with FPS=25). -1 indicates the gloss ends at the last frame of the video.

instance_id: int, id of the instance in the same class/gloss.

signer_id: int, id of the signer.

source: str, a string identifier for the source site.

split: str, indicates sample belongs to which subset.

url: str, used for video downloading.

variation_id: int, id for dialect (indexed from 0).

video_id: str, a unique video identifier.

Please be kindly advised that if you decode with different FPS, you may need to recalculate the frame_start and frame_end to get correct video segments.
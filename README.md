# Sign Language Video Generation
The aim of this repository is to generate videos in American Sign Language (ASL) from text input.

## Repository Description
### Docker image
This repository contains the Docker image for our application, located in the docker_image folder.

The algorithm converts text into a sequence of glosses and then generates a video based on these gloss sequences.

To generate videos from glosses, we use our own dataset called 'enhanced_dataset'. In this dataset, each gloss is associated with a sequence of body landmarks of the signer. We extracted these landmarks using Mediapipe from videos corresponding to each gloss. The videos were sourced from the WASL dataset.

### Python script Details
The script generate_video_landmarks.py was used to generate JSON files containing all the landmarks for each gloss (one JSON file per gloss). We applied this script to the videos from the WASL dataset (we have first downloaded locally following the procedure described in the repository containing the dataset).

## Dataset Links
Text to Gloss Translation: Hugging Face Dataset
Gloss to Video: WLASL Dataset ('https://github.com/dxli94/WLASL')
    This dataset contains URLs to video of sign language for specific glosses.



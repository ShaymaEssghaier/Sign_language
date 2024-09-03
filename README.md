# Sign Language Video Generation
The aim of this repository is to generate videos in American Sign Language (ASL) from text input.

## Repository Description
This repository contains the Docker image for our application, located in the docker_image folder.

The algorithm converts text into a sequence of glosses and then generates a video based on these gloss sequences.

To generate videos from glosses, we use our own dataset called 'enhanced_dataset'. In this dataset, each gloss is associated with a sequence of body landmarks of the signer. We extracted these landmarks using Mediapipe from videos corresponding to each gloss. The videos were sourced from the WASL dataset.

## Dataset Links
Text to Gloss Translation: Hugging Face Dataset
Gloss to Video: WLASL Dataset ('https://github.com/dxli94/WLASL')
    This dataset contains URLs to videos of sign language for specific glosses.



# Action-Recognition-for-Smart-Home

# 1. Back Pain Action Recognition model
In this project I’ll build a neural network and train it to detect back pain action in humans using dataset which consists of self-prepared videos as well as few videos collected from YouTube.

# 2. Dataset
The back pain video dataset contains 41 videos: 30 for training and 11 for testing. These videos have been captured on OnePlus 6T phone camera with specifications Sony IMX 519 with 16MP. The videos have been separated in frames per second and all the frames have been labelled as 0 when there is no action of back pain happening in the video and 1 when the person in the video is doing action of back pain. 

# 3. Trained Model
The model used for this project is VGG16 (Virtual Geometry Group with 16 layers). I have used 5 hidden layers with 2048, 1024, 512, 256 and 128 neurons respectively.

# 4. Output
The model gives output json file and a time vs label plot that shows the probability of action with respect to video.

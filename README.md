This is team IP Man's submission to AWS Hackdays 2019 -- Healthtech. 

### What it does

1) The web client sends video stream data (from the user's webcam) to a flask server using socketio
2) The server does some processing on the video stream
3) The client receives the processed video stream and re-displays the results in a different frame

In the demo site, we are using a MobileNet backend for inference of keypoints, and calculating a score based on how powerful we think your punch is. 

### Demo
[Live Demo](http://13.229.209.149/)

### Setup
Just start punching. A fast internet connection will help. # ipman

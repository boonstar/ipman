# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture(args.input if args.input else 0)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)
        
    # --------------------------------------------------------------------Sharon Start
    '''
    some easy code examples

    neck detected example
        test if neck detected first
        dynamic text label updates if neck is detected.

    code
        text_label = False
        if points[1]:
            text_label = True
        if text_label == True:
            cv.putText(frame, 'neck detected', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    face in camera example
    body_parts used:
        Neck -- 1
        Nose -- 0
        Leye -- 15
        Reye -- 14

    code
        text_label = False
        if points[1] and points[0] and points[14] and points[15]:
            text_label = True

        if text_label == True:
            cv.putText(frame, 'face in camera', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    label documentation:
    https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html
    '''

    # right punch detection
    # ----------------
    '''
    syntax
        cv2.putText(frame, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
    '''

    if points[2] and points[3] and points[4]:
        cv.putText(frame, 'Right arm is detected', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        x_coords = [x[0] for x in points[2:5]]
        y_coords = [y[1] for y in points[2:5]]

        # condition where arm is straight within a threshold of 10
        shoulder = y_coords[0]
        if all(y > shoulder-10 for y in y_coords) and all(y < shoulder+10 for y in y_coords):
            cv.putText(frame, 'Punching straight', (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            pass

        #    cv.putText(frame, 'right arm detected completely', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    else:
        cv.putText(frame, 'Right arm is not detected', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # CG detection
    # ----------------
    # first calculate hips

    # hip_x = int((points[8][0] + points[11][0])/2)
    # hip_y = int((points[8][1] + points[11][1])/2)
    # pelvis = (hip_x, hip_y)
    #
    # if pelvis and points[1]: # both pelvis and neck are visible
    #
    # --------------------------------------------------------------------Sharon End
        
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    print(Image.fromarray(frame))
    cv.imshow('OpenPose using OpenCV', frame)
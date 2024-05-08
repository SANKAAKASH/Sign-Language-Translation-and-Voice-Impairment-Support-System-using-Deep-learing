import cv2
import numpy as np
from unified_detector import Fingertips
from hand_detector.detector import SOLO, YOLO

hand_detection_method = 'yolo'

if hand_detection_method == 'solo':
    hand = SOLO(weights='weights/solo.h5', threshold=0.8)
elif hand_detection_method == 'yolo':
    hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
else:
    assert False, "'" + hand_detection_method + \
                  "' hand detection does not exist. Use either 'solo' or 'yolo' as hand detection method"

fingertips = Fingertips(weights='weights/fingertip.h5')

cam = cv2.VideoCapture(0)
print('Unified Gesture & Fingertips Detection')

# Names for the 8 signals
signal_names = [
    "Point Left",
    "Point Right",
    "Point Up",
    "Point Down",
    "Click",
    "Double Click",
    "Zoom In",
    "Zoom Out"
]

while True:
    ret, image = cam.read()

    if ret is False:
        break

    # Hand detection
    tl, br = hand.detect(image=image)

    if tl and br is not None:
        cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
        height, width, _ = cropped_image.shape

        # Gesture classification and fingertips regression
        prob, pos = fingertips.classify(image=cropped_image)
        pos = np.mean(pos, 0)

        # Post-processing
        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]

        # Drawing
        index = 0
        color = [
            (15, 15, 240),   # Point Left
            (15, 240, 155),  # Point Right
            (240, 155, 15),  # Point Up
            (240, 15, 155),  # Point Down
            (240, 15, 240),  # Click
            (0, 255, 0),     # Double Click
            (255, 0, 0),     # Zoom In
            (0, 0, 255)      # Zoom Out
        ]
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
        for c, p in enumerate(prob):
            if p > 0.5:
                signal_name = signal_names[c]
                image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12,
                                   color=color[c], thickness=-2)
                cv2.putText(image, signal_name, (int(pos[index]), int(pos[index + 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[c], 2)
            index = index + 2

    if cv2.waitKey(1) & 0xff == 27:
        break

    # Display image
    cv2.imshow('Unified Gesture & Fingertips Detection', image)

cam.release()
cv2.destroyAllWindows()

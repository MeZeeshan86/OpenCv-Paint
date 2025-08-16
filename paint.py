import cv2
import numpy as np
import os
import mediapipe as mp


# Create a blank canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)


# Load images from the Paint folder
folder_path = "images"
mylist = os.listdir(folder_path)
header = []


for imgpath in mylist:
    image= cv2.imread(os.path.join(folder_path, imgpath))
    header.append(image)
top=header[0]


tipids=[4, 8, 12, 16, 20]
xp,yp=0,0
brushThickness=10
eraserThickness=50
drawcolor=(0, 0, 0)


# Create a blank canvas for drawing
canvas= np.zeros((720, 1280, 3), dtype=np.uint8)



# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands= mp_hands.Hands(min_detection_confidence=0.85,max_num_hands=1)


# Initialize webcam
cap= cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirror effect
    img = cv2.flip(img, 1)

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results = hands.process(img_rgb)
    

    if results.multi_hand_landmarks:
        for hands_lms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hands_lms, mp_hands.HAND_CONNECTIONS)

            # Initialize finger count
            fingers_count=[]

            # # Get the coordinates of the index finger tip
            index_finger_tip = hands_lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x1 = int(index_finger_tip.x * img.shape[1])
            y1 = int(index_finger_tip.y * img.shape[0])
            
            # #Get the coordinates of the index finger tip
            middle_finger_tip = hands_lms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            x2 = int(middle_finger_tip.x * img.shape[1])
            y2 = int(middle_finger_tip.y * img.shape[0])

            #Thumb
            if hands_lms.landmark[tipids[0]].x < hands_lms.landmark[tipids[0] - 1].x:
                #cv2.circle(img, (int(handlms.landmark[tipids[0]].x * w), int(handlms.landmark[tipids[0]].y * h)), 15, (0, 255, 0), cv2.FILLED)
                fingers_count.append(1)
            else:
                #cv2.circle(img, (int(handlms.landmark[tipids[0]].x * w), int(handlms.landmark[tipids[0]].y * h)), 15, (255, 0, 0), cv2.FILLED)
                fingers_count.append(0)

            # Check if the index finger and middle finger
            for id in range(1, 5):
                if hands_lms.landmark[tipids[id]].y < hands_lms.landmark[tipids[id] - 2].y:
                     #cv2.circle(img, (int(handlms.landmark[tipids[id]].x * w), int(handlms.landmark[tipids[id]].y * h)), 15, (0, 255, 0), cv2.FILLED)
                    fingers_count.append(1)
                else:
                     #cv2.circle(img, (int(handlms.landmark[tipids[id]].x * w), int(handlms.landmark[tipids[id]].y * h)), 15, (255, 0, 0), cv2.FILLED)
                    fingers_count.append(0)


            #Selection Mode
            if fingers_count[1]==1 and fingers_count[2]==1:
                #print("Selection Mode")
                #Dawing a rectangle around the index finger tip and middle finger tip
                cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), (0, 255, 0), 2, cv2.FILLED )
                if y1 < 125:
                    # If the index finger is in the header area, change the canvas
                    if 0 < x1 < 320:
                        top = header[1]
                        drawcolor = (0, 0, 255)#red
                    elif 320 <= x1 < 640:
                        top = header[2]
                        drawcolor = (255, 0, 0)#blue
                    elif 640 <= x1 < 960:
                        top = header[3]
                        drawcolor = (0, 255, 0)#green
                    elif 960<= x1 < 1280:
                        top = header[4]
                        drawcolor=(0, 0, 0)#eraser
                    xp,yp=0,0


            #Drawing Mode
            if fingers_count[1]==1 and fingers_count[2]==0:
                #print("Drawing Mode")
                # Draw a circle at the index finger tip position
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                if xp==0 and yp==0:
                    xp, yp = x1, y1
                if drawcolor == (0, 0, 0):
                    cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, eraserThickness)
                else:
                    cv2.line(canvas, (xp, yp), (x1, y1), drawcolor, brushThickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0


    # Combine the canvas with the original image      
    imggray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imginv = cv2.threshold(imggray, 20, 255, cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imginv)
    img = cv2.bitwise_or(img, canvas)

    # Display the header image at the top of the frame
    img[0:125, 0:1280] = top
    #Closing the Frame
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
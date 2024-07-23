import cv2
import numpy as np
import imutils
import time
from scipy.signal import find_peaks
from camera import Camera
from openai import OpenAI
import openai   
import os

def sendMessage(count, bool):
    if bool:
        openai.api_key = os.getenv('OPEN_API_KEY)')
    response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "system", "content": "Only respond in on senetence. Your job is to tell how good I am at juggling a soccer ball with my feet. I will tell you how many times I just juggled the ball and you will give me feedback. Try and be funny and you can be mean. Include the juggle count early in your response"},
    {"role": "user", "content": f"I just juggled the ball {count} times."}
  ],
    )
    return (response.choices[0].message.content)
    


# Filter a specified color in an image
def ball_finder(frame, hsv_lower, hsv_upper):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

# Count the number of peaks in a time series
def peak_counter(center_):
    y_coordinates = [i[1] for i in center_]
    y_inv = np.array(y_coordinates) * (-1)
    peaks, _ = find_peaks(y_inv, height=(-200, 0), distance=12)
    return len(peaks)

# Set the parameters
blueLower = np.array([100, 50, 50])
blueUpper = np.array([140, 255, 255])

boo = True

center_ = []

camera = Camera()

# Initialize variables to track the time since the last peak
last_peak_time = time.time()
last_detection_time = time.time()  
peak_reset_time = 3  
detection_reset_time = 3  

counter = 0

chat = ""

tries =10
while tries != 0:
    time.sleep(0.025)
    
    ret, frame = camera.getframe()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)
    mask = ball_finder(frame, blueLower, blueUpper)
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_time = time.time()

    if cnts:
        last_detection_time = current_time  # Update last detection time

        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        center_.append(center)

        num_peaks = peak_counter(center_)
        if num_peaks > 0:
            last_peak_time = current_time  # Reset the last peak time
            counter = num_peaks
        if current_time - last_peak_time > detection_reset_time:
            # If ball hasn't been detected for a while, reset the counter
            counter = 0
            center_ = []  # Clear the center_ list

    else:
        if current_time - last_detection_time > detection_reset_time and counter != 0:
            # If ball hasn't been detected for a while, reset the counter
            chat = sendMessage(counter,boo)
            boo = False
            counter = 0
            center_ = []  # Clear the center_ list
            tries -= 1

    if current_time - last_peak_time > peak_reset_time:
        counter = 0
        center_ = []  # Clear the center_ list
        last_peak_time = current_time  # Reset the timer

    cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    if chat != "":
        position = (50, 150)  
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.5
        font_thickness = 1

        text_color = (0, 255, 0)  # Green
        background_color = (0, 0, 0)  # Black for contrast

        (text_width, text_height), baseline = cv2.getTextSize(chat, font, font_scale, font_thickness)

        position = (position[0], position[1] + text_height)

        cv2.rectangle(frame, (position[0], position[1] - text_height), (position[0] + text_width, position[1] + baseline), background_color, -1)

        cv2.putText(frame, chat, position, font, font_scale, text_color, font_thickness)

            # Draw the contours if any
    if cnts:
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

    cv2.imshow("Juggler Counter", frame)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 13:  # Enter key
        break

camera.release()
cv2.destroyAllWindows()

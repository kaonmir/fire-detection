import cv2
import numpy as np
import playsound
import threading
import smtplib
import os

# Environment variables
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# Validate environment variables
if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
    raise ValueError("Please set the required environment variables.")


Alarm_Status = False
Email_Status = False
Fire_Reported = 0
frame_check = 10


def play_alarm_sound_function():
    while True:
        playsound.playsound("alarm-sound.mp3", True)


def send_mail_function():
    recipientEmail = RECIPIENT_EMAIL
    recipientEmail = recipientEmail.lower()

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(
            SENDER_EMAIL,
            recipientEmail,
            "Subject: Fire Alert!\n\nWarning! A fire has been detected. Please check your security camera footage.",
        )
        print("sent to {}".format(recipientEmail))
        server.close()
    except Exception as e:
        print(e)


video = cv2.VideoCapture("fire.mp4")
first_frame = None

while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break

    frame = cv2.resize(frame, (960, 540))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    movement = np.sum(thresh) / 255

    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame, hsv, mask=mask)
    no_red = cv2.countNonZero(mask)

    if int(no_red) > 15000 and movement > frame_check:
        Fire_Reported = Fire_Reported + 1

    cv2.imshow("output", output)

    if Fire_Reported >= 1:
        if not Alarm_Status:
            threading.Thread(target=play_alarm_sound_function).start()
            Alarm_Status = True
        if not Email_Status:
            threading.Thread(target=send_mail_function).start()
            Email_Status = True

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
video.release()

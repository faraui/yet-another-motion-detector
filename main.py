import cv2
import datetime

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 6)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while camera.isOpened():
    r, frame = camera.read()
    r, newFrame = camera.read()

    difference = cv2.absdiff(frame, newFrame)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 14, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(threshold, None, iterations=3)
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 640:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+y), (255, 0, 0), 1)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        cv2.imwrite(f'F_{timestamp}.jpg', frame)
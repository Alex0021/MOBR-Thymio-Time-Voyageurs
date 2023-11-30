import cv2

# Open camera feed & initialize map
wait_for_map = True
cv2.namedWindow("Thymio Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thymio Tracking", 1680, 1050)
cam_feed = cv2.VideoCapture(0)
cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 1680)
cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 1050)
cam_feed.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'mjpg'))
print("Cam codec: ", cam_feed.get(cv2.CAP_PROP_FOURCC))
h = int(cam_feed.get(cv2.CAP_PROP_FOURCC))
codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
print(codec)
if not cam_feed.isOpened():
    print("ERROR :: Could not open camera video feed!")
while (wait_for_map):
    ret, frame = cam_feed.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Undistord image
    cv2.imshow("Thymio Tracking",frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        quit = True
        wait_for_map = False
    elif key == ord(' '):
        # Create map
        pass


cv2.destroyAllWindows()
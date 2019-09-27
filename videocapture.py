import cv2 as cv
from time import gmtime, strftime

# Load the model
net = cv.dnn.readNet('face-detection-adas-0001.xml', 'face-detection-adas-0001.bin')

# Specify target device
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

cap = cv.VideoCapture(0)

counter = 0
# Draw detected faces on the frame
while(True):
  ret, frame = cap.read()
  # Prepare input blob and perform an inference
  blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
  net.setInput(blob)
  out = net.forward()


  for detection in out.reshape(-1, 7):
      confidence = float(detection[2])
      xmin = int(detection[3] * frame.shape[1])
      ymin = int(detection[4] * frame.shape[0])
      xmax = int(detection[5] * frame.shape[1])
      ymax = int(detection[6] * frame.shape[0])

      if confidence > 0.5:
          cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
          counter = counter + 1
          if counter > 100:
              save_name = "img_" + strftime("%Y-%m-%d_%H%M%S") + ".jpg"
              cv.imwrite(save_name, frame)
              counter = 0


  cv.imshow("Video", frame)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()
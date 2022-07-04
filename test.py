import cv2
import torch



imagesx = cv2.imread('bi_86.jpg')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt')

def det(images):
    im2 = images[..., ::-1]  # OpenCV image (BGR to RGB)
    imgs = [im2]
    results = model(imgs, size=640)  # includes NMS
    x = []
    for i in range(0,len(results.pandas().xyxy[0])):
        listem = [i,int(results.pandas().xyxy[0].xmin[i]),
        int(results.pandas().xyxy[0].ymin[i]),
        int(results.pandas().xyxy[0].xmax[i]),
        int(results.pandas().xyxy[0].ymax[i]),
        round(results.pandas().xyxy[0].confidence[i],2),
        results.pandas().xyxy[0].name[i]]
        x.append(listem)

    for a in x:
        cv2.rectangle(images, (a[1],a[2]), (a[3],a[4]), (0, 255, 0), 2)
    return images

cap = cv2.VideoCapture("output.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    im = det(frame)
    cv2.imshow('image', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

import cv2
import torch
import numpy as np
import urllib

path = f'/Users/brunaoliveira/test_drone/best.pt' # (OPCIONAL) TROQUE PELO CAMINHO DO SEU PESO CASO QUEIRA (best.pt que foi gerado no treinamento) ex: yolov5/runs/train/exp9/weights/best.pt
image_url = 'http://192.168.1.100/cam-hi.jpg' # TROQUE PELO LINK GERADO NO MONITOR SERIAL

model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

print(path)

while True:
    img_resp=urllib.request.urlopen(url=image_url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    im = cv2.imdecode(imgnp,-1)

    results = model(im)

    print(results)

    frame = np.squeeze(results.render())

    cv2.imshow('Deteccao', frame)

    key=cv2.waitKey(5)

    if key==ord('q'):
        break

cv2.destroyAllWindows()
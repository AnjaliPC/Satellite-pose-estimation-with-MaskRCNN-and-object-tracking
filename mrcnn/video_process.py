import cv2
import numpy as np
import matplotlib.pyplot as plt
#from visualize_cv2 import model, display_instances, class_names


#====TRIED===IN===GOOGLE====COLAB====#
#capture = cv2.VideoCapture('/content/Mask_RCNN/2021-10-29_124554_000-Copy1.avi')
#====TRIED===IN===GOOGLE====COLAB====#
capture = cv2.VideoCapture('C:/Users/user/Desktop/EVERYTHING/APC/Mtech/Mtech-Sem3/DISSERTATION/FINAL_DATASET/Video_2.avi')

size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('C:/Users/user/Desktop/EVERYTHING/APC/Mtech/Mtech-Sem3/DISSERTATION/FINAL_DATASET/project/MaskRCNN/Masked_Video_2.avi', codec, 60.0, size)

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        output.write(frame)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()
        # cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()
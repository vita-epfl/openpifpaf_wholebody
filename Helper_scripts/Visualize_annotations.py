import json
import cv2
import numpy as np

json_file = "../data-mscoco/annotations/person_keypoints_val2017_wholebody_pifpaf_style.json"
image_dir = "../data-mscoco/images/val2017/"
color = (0,255,0)

with open(json_file, 'r') as f:
        val_data = json.load(f)

print("Press q to quit, press any other key to see the next image.")
for ann_dict in val_data["annotations"]:
    image = cv2.imread(image_dir + str(ann_dict["image_id"]).zfill(12)+".jpg")
    kps = np.round(np.array(ann_dict["keypoints"])).astype(int)
    x = kps[0::3]
    y = kps[1::3]
    for count, coordinates in enumerate(zip(x,y)):
        cv2.putText(image, str(count+1), coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.25, color)
        cv2.drawMarker(image, coordinates, color=color, markerType=cv2.MARKER_CROSS, thickness=1, markerSize=7)
    cv2.imshow("img", image)
    k=cv2.waitKey(0)
    if k==ord('q'): #break if q is pressed
        break
cv2.destroyAllWindows()
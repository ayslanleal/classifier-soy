import cv2
import glob
import os

no_of_classes = 7
insect_images = []
labels = []
i = 0
for insect_dir_path in glob.glob("./soybean-dataset/*"):
    insect_label = insect_dir_path.split("/")[-1]
    if no_of_classes == i:
        break
    for image_path in glob.glob(os.path.join(insect_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (64, 64))
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        insect_images.append(image)
        labels.append(insect_label)
    i = i + 1

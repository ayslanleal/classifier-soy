import cv2

img = cv2.imread('./Ácaros-Verde/image4.jpg', cv2.IMREAD_COLOR)

image = cv2.resize(img, (64, 64))
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
print(image.shape)
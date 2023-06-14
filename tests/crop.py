
import cv2
image_path = 'dog.jpg'
image = cv2.imread(image_path)

# Perform object detection using YOLOv5 (code not provided)

# Assume you have the bounding box coordinates of the detected object
x, y, w, h = x[0], 100, 200, 200  # Example bounding box coordinates

# Crop the image based on the bounding box coordinates
cropped_image = image[y:y+h, x:x+w]

# Display the cropped image
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
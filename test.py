import cv2
import numpy as np

img = np.full((300, 300, 3), 255, dtype=np.uint8)
cv2.putText(img, 'Hello, Ashwin!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

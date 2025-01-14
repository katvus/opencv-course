import cv2
import numpy as np

image = cv2.imread('images//image.jpg')
window_name = 'image'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 1)перевести в grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('images//gray.jpg', gray)

# 2)отразить изображение по правой границе
turn_image = cv2.flip(image, 1)
cv2.imwrite('images//turn.jpg', turn_image)

# 3)повернуть изображение на 30 градусов вокруг заданной точки
(h, w, d) = image.shape
center = (3 * w // 4, h // 2)
M = cv2.getRotationMatrix2D(center, 30, 1.0)
rotate_image = cv2.warpAffine(image, M, (w, h))
cv2.imwrite('images//rotate.jpg', rotate_image)

# 4)сделать бинаризацию изображения
_, bin_image = cv2.threshold(gray, 130, 255, 0)
cv2.imwrite('images//binary.jpg', bin_image)

# 5)найти контуры на бинаризированном изображении
contours, _ = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_image = cv2.cvtColor(bin_image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
cv2.imwrite('images//contours.jpg', contour_image)

# 6)сделать размытие изображения
blur_image = cv2.GaussianBlur(image, (51, 51), 7)
cv2.imwrite('images//blur.jpg', blur_image)

# 7)найти canny edges на изображении
edges_image = cv2.Canny(image, 75, 255)
cv2.imwrite('images//canny_edge.jpg', edges_image)

# 8)сместить изображение на 10 пикселей вправо
translation_matrix = np.float32([[1, 0, 10], [0, 1, 0]])
offset_image = cv2.warpAffine(image, translation_matrix, (w, h))
cv2.imwrite('images//offset.jpg', offset_image)

# 9)применить операцию эрозии к изображению
kernel = np.ones((25, 25), np.uint8)
eroded_image = cv2.erode(image, kernel)
cv2.imwrite('images//eroded.jpg', eroded_image)

# 10)применить операцию диляции к изображению
dilated_image = cv2.dilate(image, kernel)
cv2.imwrite('images//dilate.jpg', dilated_image)

cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()


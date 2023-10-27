import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = "yol.jpg"
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()

# Beyaz ve sarı renk aralıklarını tanımla
lower_white = np.array([200, 200, 200], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)

lower_yellow = np.array([150, 150, 0], dtype=np.uint8)
upper_yellow = np.array([255, 255, 150], dtype=np.uint8)

# Beyaz ve sarı renkleri içeren maskeleri oluştur
white_mask = cv2.inRange(img, lower_white, upper_white)
yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)

# Yalnızca beyaz ve sarı renkleri içeren maskeyi birleştir
combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

plt.imshow(combined_mask, cmap='gray')
plt.show()

# Kenarları algıla
edge_image = cv2.Canny(combined_mask, 60, 120)

plt.imshow(edge_image, cmap='gray')
plt.show()

# Şeritleri bul
lines = cv2.HoughLinesP(edge_image, 2, np.pi / 180, 20, np.array([]), minLineLength=60, maxLineGap=120)

if lines is not None:
    zeros = np.zeros_like(img)

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Sarı ve beyaz şeritlerin üzerine mavi çizgi çiz
            cv2.line(zeros, (x1, y1), (x2, y2), (0, 0, 255), 4)

    img = cv2.addWeighted(img, 0.9, zeros, 1.0, 0.)

    plt.imshow(img)
    plt.show()

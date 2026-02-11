import cv2
import numpy as np

# ---------- Load image ----------
img = cv2.imread("images10.jpg")
img = cv2.resize(img, (640, 480))

# ---------- Color spaces ----------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

h, s, v = cv2.split(hsv)
l, a, b = cv2.split(lab)

# ---------- Illumination correction ----------
v_blur = cv2.GaussianBlur(v, (51, 51), 0)
v_corrected = cv2.divide(v, v_blur, scale=255)

# ---------- UV-blue coating emphasis ----------
blue_score = cv2.subtract(b, a)  # LAB: blue-yellow minus green-red
blue_score = cv2.normalize(blue_score, None, 0, 255, cv2.NORM_MINMAX)

# ---------- Noise reduction ----------
filtered = cv2.bilateralFilter(blue_score, 9, 75, 75)

# ---------- Thresholding ----------
_, coating_mask = cv2.threshold(
    filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# ---------- Morphology ----------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
coating_mask = cv2.morphologyEx(coating_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
coating_mask = cv2.morphologyEx(coating_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# ---------- PCB isolation ----------
contours, _ = cv2.findContours(coating_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
pcb_contour = max(contours, key=cv2.contourArea)

pcb_mask = np.zeros_like(coating_mask)
cv2.drawContours(pcb_mask, [pcb_contour], -1, 255, -1)

pcb_area = cv2.countNonZero(pcb_mask)

# ---------- Void detection ----------
void_mask = cv2.bitwise_and(pcb_mask, cv2.bitwise_not(coating_mask))
void_area = cv2.countNonZero(void_mask)

# ---------- Crack detection ----------
edges = cv2.Canny(v_corrected, 40, 120)
edges = cv2.bitwise_and(edges, pcb_mask)

# ---------- Bubble detection (advanced) ----------
bubble_count = 0
bubble_confidence = 0

contours, _ = cv2.findContours(coating_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if 40 < area < 800:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity > 0.65:
            bubble_count += 1
            bubble_confidence += circularity
            cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)

# ---------- Thickness uniformity ----------
thickness_map = cv2.GaussianBlur(blue_score, (21, 21), 0)
thickness_std = np.std(thickness_map[pcb_mask > 0])

# ---------- Metrics ----------
coverage = (pcb_area - void_area) / pcb_area * 100
void_ratio = void_area / pcb_area * 100
crack_density = cv2.countNonZero(edges) / pcb_area
avg_bubble_conf = bubble_confidence / max(bubble_count, 1)

# ---------- IPC-style decision ----------
if coverage > 96 and void_ratio < 1.5 and bubble_count < 5 and thickness_std < 18:
    quality = "IPC CLASS 3 – ACCEPT"
elif coverage > 92:
    quality = "IPC CLASS 2 – CONDITIONAL"
else:
    quality = "REJECT"

# ---------- Output ----------
print(f"Coverage (%): {coverage:.2f}")
print(f"Void ratio (%): {void_ratio:.2f}")
print(f"Bubbles: {bubble_count}")
print(f"Avg bubble confidence: {avg_bubble_conf:.2f}")
print(f"Crack density: {crack_density:.4f}")
print(f"Thickness non-uniformity (σ): {thickness_std:.2f}")
print(f"Final decision: {quality}")

cv2.imshow("UV coating mask", coating_mask)
cv2.imshow("Detected defects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
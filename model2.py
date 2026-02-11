import cv2
import numpy as np

# ---------- Load & normalize image ----------
img = cv2.imread("image10.jpg")
img = cv2.resize(img, (640, 480))

# ---------- Preprocessing ----------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
v = hsv[:, :, 2]

# Reduce noise but keep edges
blur = cv2.bilateralFilter(v, 9, 75, 75)

# ---------- Contrast enhancement ----------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blur)

# ---------- Coating segmentation ----------
# Otsu is more stable than adaptive threshold here
_, coating_mask = cv2.threshold(
    enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# ---------- Morphological refinement ----------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
coating_mask = cv2.morphologyEx(coating_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
coating_mask = cv2.morphologyEx(coating_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# ---------- PCB mask (largest connected component) ----------
contours, _ = cv2.findContours(coating_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
pcb_contour = max(contours, key=cv2.contourArea)
pcb_mask = np.zeros_like(coating_mask)
cv2.drawContours(pcb_mask, [pcb_contour], -1, 255, -1)

pcb_area = cv2.countNonZero(pcb_mask)

# ---------- Void detection ----------
void_mask = cv2.bitwise_and(pcb_mask, cv2.bitwise_not(coating_mask))
void_area = cv2.countNonZero(void_mask)

# ---------- Crack detection ----------
edges = cv2.Canny(enhanced, 40, 120)
edges = cv2.bitwise_and(edges, pcb_mask)

# ---------- Bubble detection (circular defects) ----------
bubble_count = 0
bubble_areas = []

contours, _ = cv2.findContours(coating_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if 30 < area < 600:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Circular shapes → bubbles
        if circularity > 0.6:
            bubble_count += 1
            bubble_areas.append(area)
            cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)

# ---------- Quality metrics ----------
coverage = (pcb_area - void_area) / pcb_area * 100
void_ratio = void_area / pcb_area * 100
crack_density = cv2.countNonZero(edges) / pcb_area

# ---------- Final assessment ----------
if coverage > 95 and void_ratio < 2 and bubble_count < 10:
    quality = "GOOD"
elif coverage > 90:
    quality = "ACCEPTABLE"
else:
    quality = "REJECT"

# ---------- Results ----------
print(f"Coverage (%): {coverage:.2f}")
print(f"Void ratio (%): {void_ratio:.2f}")
print(f"Bubble count: {bubble_count}")
print(f"Crack density: {crack_density:.4f}")
print(f"Overall coating quality: {quality}")

# ---------- Optional visualization ----------
cv2.imshow("Original", img)
cv2.imshow("Coating mask", coating_mask)
cv2.imshow("Void mask", void_mask)
cv2.imshow("Cracks", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
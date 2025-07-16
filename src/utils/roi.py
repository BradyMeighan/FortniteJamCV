import cv2

# Load image
img = cv2.imread("screenshot.png")
clone = img.copy()
roi = None
drawing = False
ix, iy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_display = clone.copy()
        cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Draw ROI", img_display)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        cv2.rectangle(clone, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
        cv2.imshow("Draw ROI", clone)
        print(f"[ROI] x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

        # Save cropped ROI
        cropped = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        cv2.imwrite("custom_shape_roi.png", cropped)
        print("[Saved] custom_shape_roi.png")

cv2.namedWindow("Draw ROI")
cv2.setMouseCallback("Draw ROI", draw_rectangle)
cv2.imshow("Draw ROI", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# 👉 여기에 이미지 경로 넣으세요
IMG_PATH = r"C:\Users\hjy02\IdeaProjects\A-LAW-ML\data\계약서양식\images\court_lease_page0.png"

def load_image_unicode(path):
    """한글 경로/파일명에서도 안전하게 이미지 로드"""
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

img = load_image_unicode(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {IMG_PATH}")

clone = img.copy()

start_point = None
drawing = False

def mouse_callback(event, x, y, flags, param):
    global start_point, drawing, img, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        print(f"[START] x={x}, y={y}")

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img = clone.copy()
        cv2.rectangle(img, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Field Picker", img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = start_point
        x2, y2 = x, y
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        print(f"[END]   x={x}, y={y}")
        print(f"👉 FieldDefinition 좌표 = ({x1}, {y1}, {w}, {h})\n")

        cv2.rectangle(img, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Field Picker", img)
        clone = img.copy()

scale = 0.5 # 80% 크기로 줄이기

cv2.namedWindow("Field Picker", cv2.WINDOW_NORMAL)
h, w = img.shape[:2]
cv2.resizeWindow("Field Picker", int(w*scale), int(h*scale))
cv2.setMouseCallback("Field Picker", mouse_callback)
cv2.imshow("Field Picker", img)

print("📌 마우스로 드래그해서 필드를 지정하세요.")
print("   왼쪽 버튼 누르고 → 드래그 → 놓기")
print("   콘솔에 (x, y, w, h) 자동 출력됩니다.")

cv2.waitKey(0)
cv2.destroyAllWindows()

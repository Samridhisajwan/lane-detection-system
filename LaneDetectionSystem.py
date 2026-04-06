import cv2
import numpy as np

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
    [(200, height), (1100, height), (700, 400), (500,400)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        mid_x = (x1 + x2) / 2

        # Ignore almost horizontal lines
        if abs(slope) < 0.3:
            continue

        # left side of screen
        if slope < 0 and mid_x < image.shape[1] / 2:
            left_fit.append((slope, intercept))

    # right side of screen
        elif slope > 0 and mid_x > image.shape[1] / 2:
            right_fit.append((slope, intercept))
    left_avg = np.average(left_fit, axis=0) if len(left_fit) > 0 else None
    right_avg = np.average(right_fit, axis=0) if len(right_fit) > 0 else None

    return left_avg, right_avg


def make_points(image, line):
    if line is None:
        return None

    height = image.shape[0]
    y1 = height
    y2 = int(height * 0.6)

    slope, intercept = line

    if slope == 0:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

def draw_lines(image, left_line, right_line):
    line_image = np.zeros_like(image)

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(line_image,
                 (int(x1), int(y1)),
                 (int(x2), int(y2)),
                 (255, 0, 0), 5)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(line_image,
                 (int(x1), int(y1)),
                 (int(x2), int(y2)),
                 (0, 0, 255), 5)

    return line_image

def get_steering_direction(image, left_line, right_line):
    height, width, _ = image.shape

    if left_line is None and right_line is None:
        return "No Lane"

    # If one lane missing, assume center
    if left_line is None:
        return "Steer Left"
    if right_line is None:
        return "Steer Right"

    left_x2 = left_line[2]
    right_x2 = right_line[2]

    lane_center = (left_x2 + right_x2) / 2
    frame_center = width / 2

    deviation = lane_center - frame_center

    if deviation > 80:
        return "Steer Right"
    elif deviation < -80:
        return "Steer Left"
    else:
        return "Go Straight"

cap = cv2.VideoCapture(r"C:\Users\samis\Downloads\856923-hd_1920_1080_30fps.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# white color mask
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    blur = cv2.GaussianBlur(mask, (5, 5), 0)

    canny = cv2.Canny(mask, 50, 150)
    cropped = region_of_interest(canny)

    lines = cv2.HoughLinesP(
    cropped,
    1,
    np.pi / 180,
    30,
    minLineLength=50,
    maxLineGap=100
)

    left_avg, right_avg = average_slope_intercept(frame, lines)

    left_line = make_points(frame, left_avg) if left_avg is not None else None
    right_line = make_points(frame, right_avg) if right_avg is not None else None

    line_image = draw_lines(frame, left_line, right_line)

    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    direction = get_steering_direction(frame, left_line, right_line)

    cv2.putText(combo, direction, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Smart Lane Detection", combo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
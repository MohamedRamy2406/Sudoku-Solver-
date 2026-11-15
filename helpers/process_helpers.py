import operator
import cv2
import numpy as np


def find_extreme_corners(polygon, limit_fn, compare_fn):
    # limit_fn is the min or max function
    # compare_fn is the np.add or np.subtract function

    # if we are trying to find bottom left corner, we know that it will have the smallest (x - y) value
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    return polygon[section][0][0], polygon[section][0][1]


def draw_extreme_corners(pts, original):
    cv2.circle(original, pts, 7, (0, 255, 0), cv2.FILLED)


def clean_helper(img):
    height, width = img.shape

    # Ignore the cell borders to remove residual grid lines
    margin_x = int(width * 0.12)
    margin_y = int(height * 0.12)

    cropped = img[
        margin_y:height - margin_y,
        margin_x:width - margin_x
    ]

    if cropped.size == 0:
        return np.zeros_like(img), False

    # Find connected white objects
    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            cropped,
            connectivity=8
        )
    )

    candidates = []

    # Label 0 represents the background
    for index in range(1, count):
        x, y, w, h, area = stats[index]

        # Reject small noise components
        if area < height * width * 0.01:
            continue

        # Reject objects that are too short
        if h < height * 0.25:
            continue

        candidates.append(index)

    # No digit-like component was found
    if not candidates:
        return np.zeros_like(img), False

    # Select the largest valid component
    index = max(
        candidates,
        key=lambda i: stats[i, cv2.CC_STAT_AREA]
    )

    x, y, w, h, area = stats[index]

    # Isolate the selected digit
    component = np.uint8(labels == index) * 255

    digit = component[y:y + h, x:x + w]

    # Center the digit inside an empty cell
    new_img = np.zeros_like(img)

    start_x = (width - w) // 2
    start_y = (height - h) // 2

    new_img[
        start_y:start_y + h,
        start_x:start_x + w
    ] = digit

    return new_img, True


def grid_line_helper(img, shape_location, length=10):
    clone = img.copy()
    # if its horizontal lines then it is shape_location 1, for vertical it is 0
    row_or_col = clone.shape[shape_location]
    # find out the distance the lines are placed
    size = row_or_col // length

    # find out an appropriate kernel
    if shape_location == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    # erode and dilate the lines
    clone = cv2.erode(clone, kernel)
    clone = cv2.dilate(clone, kernel)

    return clone


def draw_lines(img, lines):

    clone = img.copy()

    if lines is None:
        return clone

    lines = np.asarray(lines).reshape(-1, 2)

    for rho, theta in lines:

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)

        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(
            clone,
            (x1, y1),
            (x2, y2),
            255,
            4
        )

    return clone


def extract_digit(cell, contrast=20, target_height=22):

    height, width = cell.shape

    blurred = cv2.GaussianBlur(cell, (3, 3), 0)

    margin_x = int(width * 0.15)
    margin_y = int(height * 0.15)

    cropped = blurred[
        margin_y:height - margin_y,
        margin_x:width - margin_x
    ]

    if cropped.size == 0:
        return 0

    background = np.median(cropped)

    threshold = background - contrast

    binary = np.uint8(
        cropped < threshold
    ) * 255

    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    candidates = []

    for index in range(1, count):

        x, y, w, h, area = stats[index]

        if area < width * height * 0.01:
            continue

        if h < height * 0.25:
            continue

        candidates.append(index)

    if not candidates:
        return 0

    index = max(
        candidates,
        key=lambda i: stats[i, cv2.CC_STAT_AREA]
    )

    x, y, w, h, area = stats[index]

    component = np.uint8(
        labels == index
    ) * 255

    digit = component[y:y + h, x:x + w]

    scale = target_height / h

    new_width = max(1, int(round(w * scale)))
    new_height = target_height

    if new_width > 28:

        scale = 28 / w
        new_width = 28
        new_height = max(1, int(round(h * scale)))

    digit = cv2.resize(
        digit,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.zeros(
        (32, 32),
        dtype=np.uint8
    )

    offset_x = (32 - new_width) // 2
    offset_y = (32 - new_height) // 2

    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width
    ] = digit

    return canvas
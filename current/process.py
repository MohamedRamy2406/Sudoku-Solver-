import cv2
import numpy as np
import tensorflow as tf

from helpers import process_helpers


def create_grid_mask(vertical, horizontal):

    grid = cv2.add(horizontal, vertical)

    grid = cv2.adaptiveThreshold(
        grid,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        235,
        2
    )

    grid = cv2.dilate(
        grid,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)
        ),
        iterations=2
    )

    pts = cv2.HoughLines(
        grid, 0.3, np.pi / 90, 200
    )

    # Safely reject unsuccessful line detection
    if pts is None:
        raise ValueError("No grid lines detected")

    lines = process_helpers.draw_lines(grid, pts)

    mask = cv2.bitwise_not(lines)

    return mask


def get_grid_lines(img, length=10):
    horizontal = process_helpers.grid_line_helper(img, 1, length)
    vertical = process_helpers.grid_line_helper(img, 0, length)
    return vertical, horizontal


def find_contours(img, original):
    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 1000:
            continue

        perimeter = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            0.01 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        top_left = process_helpers.find_extreme_corners(
            approx, min, np.add
        )

        top_right = process_helpers.find_extreme_corners(
            approx, max, np.subtract
        )

        bot_left = process_helpers.find_extreme_corners(
            approx, min, np.subtract
        )

        bot_right = process_helpers.find_extreme_corners(
            approx, max, np.add
        )

        corners = [
            top_left,
            top_right,
            bot_right,
            bot_left
        ]

        if len(set(corners)) != 4:
            continue

        cv2.drawContours(
            original, [approx], 0, (0, 0, 255), 3
        )

        for point in corners:
            process_helpers.draw_extreme_corners(
                point, original
            )

        return corners

    return []

def warp_image(corners, original):
    # we will be warping these points
    corners = np.array(corners, dtype='float32')
    top_left, top_right, bot_right, bot_left = corners

    # find the best side width, since we will be warping into a square, height = length
    width = int(max([
        np.linalg.norm(top_right - bot_right),
        np.linalg.norm(top_left - bot_left),
        np.linalg.norm(bot_right - bot_left),
        np.linalg.norm(top_left - top_right)
    ]))

    # create an array with shows top_left, top_right, bot_left, bot_right
    mapping = np.array([[0, 0], [width - 1, 0], [width - 1, width - 1], [0, width - 1]], dtype='float32')

    matrix = cv2.getPerspectiveTransform(corners, mapping)

    return cv2.warpPerspective(original, matrix, (width, width)), matrix


def split_into_squares(warped_img):
    squares = []

    width = warped_img.shape[0] // 9

    # find each square assuming they are of the same side
    for j in range(9):
        for i in range(9):
            p1 = (i * width, j * width)  # Top left corner of a bounding box
            p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box
            squares.append(warped_img[p1[1]:p2[1], p1[0]:p2[0]])

    return squares


def clean_squares(squares):
    cleaned_squares = []
    i = 0

    for square in squares:
        new_img, is_number = process_helpers.clean_helper(square)

        if is_number:
            cleaned_squares.append(new_img)
            i += 1

        else:
            cleaned_squares.append(0)

    return cleaned_squares


def recognize_digits(squares_processed, model):
    formatted_squares = []
    positions = []

    if len(squares_processed) != 81:
        raise ValueError("Expected exactly 81 Sudoku cells")

    for index, cell in enumerate(squares_processed):

        # Skip empty cells
        if isinstance(cell, int):
            continue

        img = cv2.resize(cell, (32, 32))

        formatted_squares.append(img)
        positions.append(index)

    result = ["0"] * 81

    # Handle a completely empty Sudoku grid
    if not formatted_squares:
        return "".join(result)

    inputs = np.array(
        formatted_squares,
        dtype=np.float32
    )

    # Add grayscale channel dimension
    inputs = np.expand_dims(inputs, axis=-1)

    # Perform batch inference
    predictions = model(
        tf.convert_to_tensor(inputs),
        training=False
    )

    predicted_classes = np.argmax(
        predictions.numpy(),
        axis=1
    )

    for index, position in enumerate(positions):
        result[position] = str(
            predicted_classes[index] + 1
        )

    return "".join(result)


def draw_digits_on_warped(warped_img, solved_puzzle, squares_processed):
    width = warped_img.shape[0] // 9

    img_w_text = np.zeros_like(warped_img)

    # find each square assuming they are of the same side
    index = 0
    for j in range(9):
        for i in range(9):
            if type(squares_processed[index]) == int:
                p1 = (i * width, j * width)  # Top left corner of a bounding box
                p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box

                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                text_size, _ = cv2.getTextSize(str(solved_puzzle[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4)
                text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                cv2.putText(warped_img, str(solved_puzzle[index]),
                            text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            index += 1

    return img_w_text


def unwarp_image(img_src, img_dest, pts, time):
    pts = np.array(pts)

    height, width = img_src.shape[0], img_src.shape[1]
    pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    cv2.fillConvexPoly(img_dest, pts, 0, 16)

    dst_img = cv2.add(img_dest, warped)

    dst_img_height, dst_img_width = dst_img.shape[0], dst_img.shape[1]
    cv2.putText(dst_img, time, (dst_img_width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    return dst_img




def extract_grayscale_cells(warped):

    gray = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    cells = []

    for row in range(9):

        for col in range(9):

            y1 = row * height // 9
            y2 = (row + 1) * height // 9

            x1 = col * width // 9
            x2 = (col + 1) * width // 9

            cell = gray[y1:y2, x1:x2]

            digit = process_helpers.extract_digit(
                cell,
                contrast=20,
                target_height=22
            )

            cells.append(digit)

    return cells
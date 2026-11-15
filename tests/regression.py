import cv2
from pathlib import Path

from core import process, sudoku
from preprocessing import preprocess
from models import model_wrapper


model = model_wrapper.model_wrapper(
    None, False, None, "models/model.hdf5"
)


TESTS = {
    "test_imgs/1.jpg":
        "205007006400960020000080045"
        "980074000570802069000630057"
        "750020000060051002300400508",

    "test_imgs/sudoku.jpg":
        "800010009050807010004090700"
        "060701020508060107010502090"
        "007040600080309040300050008",

    "test_imgs/webcam_capture.png":
        "000907030092000014306210800"
        "064078059000000000850320760"
        "005062103430000670020401000"
}


output = Path("debug_final")
output.mkdir(exist_ok=True)


for filename, expected in TESTS.items():

    print("\nTesting:", filename)

    img = cv2.imread(filename)

    if img is None:
        print("FAIL: Cannot read image")
        continue

    processed = preprocess.preprocess(img)

    corners = process.find_contours(
        processed, img.copy()
    )

    if not corners:
        print("FAIL: No Sudoku detected")
        continue

    warped, matrix = process.warp_image(
        corners, img
    )

    # New digit extraction pipeline
    cleaned = process.extract_grayscale_cells(
        warped
    )

    recognized = process.recognize_digits(
        cleaned, model
    )

    print("\nRecognized Sudoku:")

    for row in range(9):
        print(
            " ".join(
                recognized[row * 9:(row + 1) * 9]
            )
        )

    errors = sum(
        a != b
        for a, b in zip(recognized, expected)
    )

    print("Incorrect cells:", errors)

    if errors != 0:

        print("FAIL: Recognition errors")

        for i in range(81):

            if recognized[i] != expected[i]:

                print(
                    f"Row {i // 9 + 1}, "
                    f"Column {i % 9 + 1}: "
                    f"expected {expected[i]}, "
                    f"got {recognized[i]}"
                )

        continue

    solved, elapsed = sudoku.solve_wrapper(
        recognized
    )

    if solved is None:
        print("FAIL: No unique solution")
        continue

    process.draw_digits_on_warped(
        warped, solved, cleaned
    )

    result = process.unwarp_image(
        warped,
        img.copy(),
        corners,
        elapsed
    )

    destination = (
        output / (Path(filename).stem + "_solved.png")
    )

    cv2.imwrite(str(destination), result)

    print("PASS: Perfect recognition")
    print("PASS: Sudoku solved")
    print("PASS: Overlay generated")
    print("Saved:", destination)
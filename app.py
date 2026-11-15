import time
import cv2

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from current import process, sudoku
from models import model_wrapper
from preprocessing import preprocess


FRAME_WIDTH = 960
FRAME_HEIGHT = 720
CACHE_LIMIT = 100

WINDOW_NAME = "Sudoku Solver"


def add_status(img, message, color=(0, 255, 0)):

    cv2.putText(
        img,
        message,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    return img


def process_frame(img, model, seen):

    result = img.copy()

    try:

        # 1. Preprocess original image
        processed = preprocess.preprocess(img)

        # 2. Detect Sudoku corners
        corners = process.find_contours(
            processed,
            img.copy()
        )

        if not corners:

            return add_status(
                result,
                "Searching for Sudoku...",
                (0, 255, 255)
            )

        # 3. Correct perspective
        warped, matrix = process.warp_image(
            corners, img
        )

        # 4. Extract standardized digit images
        cleaned = process.extract_grayscale_cells(
            warped
        )

        if len(cleaned) != 81:

            return add_status(
                result,
                "Cell extraction failed",
                (0, 0, 255)
            )

        # 5. Recognize digits using CNN
        recognized = process.recognize_digits(
            cleaned,
            model
        )

        clues = 81 - recognized.count("0")

        if clues < 17:

            return add_status(
                result,
                f"Insufficient clues: {clues}",
                (0, 255, 255)
            )

        # 6. Check solution cache
        if recognized in seen:

            solved, elapsed = seen[recognized]
            seen.move_to_end(recognized)

        else:

            # 7. Solve Sudoku
            solved, elapsed = sudoku.solve_wrapper(
                recognized
            )

            if solved is not None:

                seen[recognized] = (
                    solved, elapsed
                )

                if len(seen) > CACHE_LIMIT:
                    seen.popitem(last=False)

                print("\nSudoku recognized:")

                for row in range(9):

                    print(
                        recognized[
                            row * 9:(row + 1) * 9
                        ]
                    )

                print("\nSudoku solved!")
                print(elapsed)

        # 8. Handle invalid or ambiguous puzzles
        if solved is None:

            return add_status(
                result,
                f"No unique solution - Clues: {clues}",
                (0, 255, 255)
            )

        # 9. Draw missing digits
        process.draw_digits_on_warped(
            warped,
            solved,
            cleaned
        )

        # 10. Project solution onto camera frame
        result = process.unwarp_image(
            warped,
            img.copy(),
            corners,
            elapsed
        )

        return add_status(
            result,
            f"Sudoku solved! Clues: {clues}"
        )

    except Exception as error:

        print("Processing error:", error)

        return add_status(
            img.copy(),
            "Processing failed - retrying",
            (0, 0, 255)
        )


def main():

    print("Loading trained model...")

    my_model = model_wrapper.model_wrapper(
        None,
        False,
        None,
        "model.hdf5"
    )

    print("Model loaded successfully.")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Cannot open webcam.")
        cap.release()
        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(
        f"Camera resolution: "
        f"{actual_width}x{actual_height}"
    )

    seen = OrderedDict()

    # Run frame processing in a background thread
    executor = ThreadPoolExecutor(max_workers=1)

    future = None
    latest_result = None
    result_time = 0

    print("Webcam opened.")
    print("Press q to quit.")
    print("Press s to save a raw camera frame.")

    try:

        while True:

            success, frame = cap.read()

            if not success:

                print("ERROR: Cannot read camera frame.")
                break

            # Retrieve processed frame
            if future is not None and future.done():

                try:

                    latest_result = future.result()

                except Exception as error:

                    print("Frame error:", error)

                    latest_result = add_status(
                        frame.copy(),
                        "Processing error",
                        (0, 0, 255)
                    )

                result_time = time.monotonic()
                future = None

            # Process the newest available frame
            if future is None:

                future = executor.submit(
                    process_frame,
                    frame.copy(),
                    my_model,
                    seen
                )

            # Select frame for display
            if (
                latest_result is not None
                and time.monotonic() - result_time < 1.0
            ):

                display = latest_result

            else:

                display = add_status(
                    frame.copy(),
                    "Processing..."
                )

            # Display one webcam window
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            # Exit application
            if key == ord("q"):
                break

            # Save current raw frame
            if key == ord("s"):

                if cv2.imwrite(
                    "webcam_capture.png", frame
                ):

                    print("Camera frame saved.")

                else:

                    print("Failed to save frame.")

    except KeyboardInterrupt:

        print("\nInterrupted by user.")

    finally:

        cap.release()
        cv2.destroyAllWindows()

        executor.shutdown(wait=True)

        print("Webcam released.")
        print("Application closed.")


if __name__ == "__main__":
    main()
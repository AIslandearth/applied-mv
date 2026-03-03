"""
main.py
-------
Entry point.  Swap in your own image path and tweak LineDetector params here.
"""

from line_detector import App, LineDetector


def main() -> None:
    detector = LineDetector(
        blur_size=7,
        canny_low=200,
        canny_high=300,
        hough_threshold=80,
        min_line_length=60,
        max_line_gap=20,
        top_n=3,
    )

    app = App(
        image_path="sources/img/testImage3.png",
        detector=detector,
    )
    app.run()


if __name__ == "__main__":
    main()

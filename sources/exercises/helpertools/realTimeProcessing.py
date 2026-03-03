# ═══════════════════════════════════════════════════════
# REAL TIME VIDEO
# ═══════════════════════════════════════════════════════

def process_video(source=0, use_maze=False):
    cap     = cv2.VideoCapture(source)
    frame_n = 0
    H       = None       # stored warp matrix — reused every frame
    warped  = None       # last warped result

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Every frame — fast diff detection ───────────
        diff = Diff(gray, threshold=30, step=2)
        diff.detect()
        cv2.imshow("Diff edges", diff.draw())

        # ── Every 30 frames — full detection + warp ─────
        # Maze or Diff depending on use_maze flag
        if frame_n % 30 == 0:
            if use_maze:
                detector = Maze(gray, threshold=30, step=2)
            else:
                detector = Diff(gray, threshold=30, step=2)

            detector.detect()

            if detector.corners is not None:
                # Store H matrix — reuse next 29 frames
                dst = np.float32([[0,0],[500,0],[500,500],[0,500]])
                H   = cv2.getPerspectiveTransform(detector.corners, dst)

        # ── Every frame — apply stored H (cheap) ────────
        if H is not None:
            warped = cv2.warpPerspective(gray, H, (500, 500))

            # Detect on warped — now clean and aligned
            warped_diff = Diff(warped, threshold=30, step=2)
            warped_diff.detect()
            cv2.imshow("Warped", warped_diff.draw(warped))

        frame_n += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════
# STATIC IMAGE
# ═══════════════════════════════════════════════════════

def process_static(path, use_maze=False):
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect
    detector = Maze(gray, threshold=30, step=2) if use_maze else \
               Diff(gray, threshold=30, step=2)
    detector.detect()

    # Show original with edges
    cv2.imshow("Original", detector.draw())

    # Warp + detect on warped
    warped = detector.warp()
    if warped is not None:
        warped_detector = Maze(warped, threshold=30, step=2) if use_maze else \
                          Diff(warped, threshold=30, step=2)
        warped_detector.detect()
        cv2.imshow("Warped", warped_detector.draw(warped))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":

    # Static image
    process_static("image.png", use_maze=False)   # Diff
    process_static("image.png", use_maze=True)    # Maze

    # Real time video
    # process_video(source=0, use_maze=False)     # Diff every frame
    # process_video(source=0, use_maze=True)      # Maze every 30 frames
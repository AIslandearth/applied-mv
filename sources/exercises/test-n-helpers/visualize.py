"""
visualize.py
============
Helpers for inspecting processed arrays from StepDiff and Derivative.

Usage
-----
    from visualize import Visualizer

    vis = Visualizer(sd, title="StepDiff")
    vis.show_diff_array()       # heatmap of the binary diff array
    vis.show_row_profile(50)    # raw gray + d1 + d2 for one row
    vis.show_col_profile(80)    # same but for a column
    vis.show_points_scatter()   # scatter of all detected centers
    vis.export_csv("out.csv")   # detected points to csv
    vis.export_excel("out.xlsx")# detected points to excel
    vis.show_all(row=50, col=80)# everything in one call
"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# try:
#     import matplotlib.pyplot as plt
#     import matplotlib.gridspec as gridspec
#     HAS_MPL = True
# except ImportError:
#     HAS_MPL = False
#     print("[visualize] matplotlib not found — install with: pip install matplotlib")

# try:
#     import pandas as pd
#     HAS_PD = True
# except ImportError:
#     HAS_PD = False
#     print("[visualize] pandas not found — install with: pip install pandas")

HAS_PD = True

class Visualizer:
    """
    Wraps a StepDiff or Derivative detector instance and provides
    array visualization and export methods.

    Parameters
    ----------
    detector : StepDiff or Derivative instance (after .detect() has run)
    title    : label shown in plot titles
    """

    def __init__(self, detector, title: str = "Detector"):
        self.det   = detector
        self.title = title

    # Diff array heatmap

    def show_diff_array(self, cmap: str = "hot"):
        """
        Show the processed diff/binary array as a heatmap.
        Works for both StepDiff (binary 0/1) and Derivative (continuous d1).
        """

        arr = self._get_arr()
        if arr is None:
            print("[show_diff_array] No array found on detector.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{self.title} — diff array", fontsize=13)

        # Left: full heatmap
        im = axes[0].imshow(arr, cmap=cmap, aspect='auto')
        axes[0].set_title("Full array (heatmap)")
        axes[0].set_xlabel("x (column)")
        axes[0].set_ylabel("y (row)")
        plt.colorbar(im, ax=axes[0])

        # Right: original gray for comparison
        axes[1].imshow(self.det.gray, cmap='gray', aspect='auto')
        axes[1].set_title("Original grayscale")
        axes[1].set_xlabel("x (column)")

        plt.tight_layout()
        plt.show()

    # Single row profile

    def show_row_profile(self, row: int):
        """
        For a given row, plot:
          - raw grayscale intensity
          - d1  (first derivative)
          - d2  (second derivative)
          - detected outer / center / inner positions as vertical lines
        """

        gray    = self.det.gray
        h, w    = gray.shape
        row     = int(np.clip(row, 0, h - 1))
        profile = gray[row, :].astype(np.float32)

        smooth_k = np.array([0.25, 0.50, 0.25], dtype=np.float32)
        p  = np.convolve(profile, smooth_k, mode='same')
        d1 = np.diff(p)
        d2 = np.diff(d1)

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
        fig.suptitle(f"{self.title} — row {row} profile", fontsize=13)

        # Raw intensity
        axes[0].plot(profile, color='gray', linewidth=1)
        axes[0].set_ylabel("Intensity")
        axes[0].set_title("Grayscale intensity")
        axes[0].grid(True, alpha=0.3)

        # d1
        axes[1].plot(d1, color='steelblue', linewidth=1)
        axes[1].axhline(0, color='black', linewidth=0.5)
        axes[1].axhline( getattr(self.det, 'threshold', 15),
                         color='red', linewidth=0.8, linestyle='--',
                         label='threshold')
        axes[1].set_ylabel("d1")
        axes[1].set_title("1st derivative")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # d2
        axes[2].plot(d2, color='darkorange', linewidth=1)
        axes[2].axhline(0, color='black', linewidth=0.5)
        axes[2].set_ylabel("d2")
        axes[2].set_title("2nd derivative (zero crossings = edge tops)")
        axes[2].grid(True, alpha=0.3)

        # Mark detected points on this row
        if len(self.det.ys) > 0:
            mask = self.det.ys == row
            if mask.any():
                for ax in axes:
                    for o in self.det.outers[mask]:
                        ax.axvline(o, color='blue',  alpha=0.5,
                                   linewidth=1, linestyle=':')
                    for c in self.det.centers[mask]:
                        ax.axvline(c, color='green', alpha=0.8,
                                   linewidth=1.2, linestyle='-')
                    for i in self.det.inners[mask]:
                        ax.axvline(i, color='red',   alpha=0.5,
                                   linewidth=1, linestyle=':')
                axes[0].legend(
                    handles=[
                        plt.Line2D([0],[0], color='blue',  label='outer'),
                        plt.Line2D([0],[0], color='green', label='center'),
                        plt.Line2D([0],[0], color='red',   label='inner'),
                    ],
                    fontsize=8,
                )

        for ax in axes:
            ax.set_xlabel("x (column)")

        plt.tight_layout()
        plt.show()

    # Single column profile

    def show_col_profile(self, col: int):
        """Same as show_row_profile but for a vertical column scan."""

        gray    = self.det.gray
        h, w    = gray.shape
        col     = int(np.clip(col, 0, w - 1))
        profile = gray[:, col].astype(np.float32)

        smooth_k = np.array([0.25, 0.50, 0.25], dtype=np.float32)
        p  = np.convolve(profile, smooth_k, mode='same')
        d1 = np.diff(p)
        d2 = np.diff(d1)

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
        fig.suptitle(f"{self.title} — column {col} profile", fontsize=13)

        axes[0].plot(profile, color='gray',       linewidth=1)
        axes[0].set_title("Grayscale intensity")
        axes[0].set_ylabel("Intensity")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(d1, color='steelblue',       linewidth=1)
        axes[1].axhline(0, color='black',         linewidth=0.5)
        axes[1].axhline(getattr(self.det, 'threshold', 15),
                        color='red', linewidth=0.8, linestyle='--',
                        label='threshold')
        axes[1].set_title("1st derivative")
        axes[1].set_ylabel("d1")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(d2, color='darkorange',      linewidth=1)
        axes[2].axhline(0, color='black',         linewidth=0.5)
        axes[2].set_title("2nd derivative")
        axes[2].set_ylabel("d2")
        axes[2].grid(True, alpha=0.3)

        for ax in axes:
            ax.set_xlabel("y (row)")

        plt.tight_layout()
        plt.show()

    # Scatter of all detected centers

    def show_points_scatter(self):
        """
        Scatter plot of all detected center positions overlaid on
        the original grayscale image.
        """

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.det.gray, cmap='gray', aspect='auto')

        if len(self.det.ys) > 0:
            ax.scatter(self.det.centers, self.det.ys,
                       s=1, c='red', alpha=0.5, label='centers')
            ax.scatter(self.det.outers,  self.det.ys,
                       s=1, c='blue',  alpha=0.3, label='outers')
            ax.scatter(self.det.inners,  self.det.ys,
                       s=1, c='lime',  alpha=0.3, label='inners')

        ax.set_title(f"{self.title} — detected points ({len(self.det.ys)} total)")
        ax.legend(markerscale=6, fontsize=9)
        ax.set_xlabel("x (column)")
        ax.set_ylabel("y (row)")
        plt.tight_layout()
        plt.show()

    # Width histogram

    def show_width_histogram(self, bins: int = 40):
        """Histogram of detected edge widths — useful for tuning threshold."""
        if len(self.det.widths) == 0:
            print("[show_width_histogram] No points detected.")
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(self.det.widths, bins=bins, color='steelblue', edgecolor='white')
        ax.set_title(f"{self.title} — edge width distribution")
        ax.set_xlabel("width (px)")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Print summary to console

    def print_summary(self):
        """Print a compact numerical summary to console."""
        n = len(self.det.ys)
        print(f"\n{'─'*50}")
        print(f"  {self.title} — summary")
        print(f"{'─'*50}")
        print(f"  Detected points : {n}")
        if n > 0:
            print(f"  y   range       : {self.det.ys.min()} → {self.det.ys.max()}")
            print(f"  center range    : {self.det.centers.min()} → {self.det.centers.max()}")
            print(f"  width  mean/max : {self.det.widths.mean():.1f} / {self.det.widths.max()}")
        print(f"  Image shape     : {self.det.gray.shape}")
        print(f"  Threshold       : {self.det.threshold}")
        print(f"  Step            : {self.det.step}")
        arr = self._get_arr()
        if arr is not None:
            print(f"  Diff arr shape  : {arr.shape}")
            print(f"  Diff arr dtype  : {arr.dtype}")
            print(f"  Diff arr min/max: {arr.min()} / {arr.max()}")
        print(f"{'─'*50}\n")

    # Print a small patch of the diff array

    def print_patch(self, row: int, col: int, size: int = 20):
        """
        Print a size×size patch of the diff array to console as integers.
        Useful for spot-checking a specific region.
        """
        arr = self._get_arr()
        if arr is None:
            print("[print_patch] No array available.")
            return
        h, w = arr.shape
        r0, r1 = max(0, row - size//2), min(h, row + size//2)
        c0, c1 = max(0, col - size//2), min(w, col + size//2)
        patch  = arr[r0:r1, c0:c1]
        print(f"\nDiff array patch  rows {r0}:{r1}  cols {c0}:{c1}")
        print(np.array2string(patch.astype(np.int16),
                              max_line_width=120,
                              separator=' '))

    def export_csv(self, path: str = "points.csv"):
        """Export detected points (y, outer, center, inner, width) to CSV."""
        if not HAS_PD:
            # fallback: plain numpy
            data = np.column_stack([
                self.det.ys,
                self.det.outers,
                self.det.centers,
                self.det.inners,
                self.det.widths,
            ])
            np.savetxt(path, data, fmt='%d',
                       delimiter=',',
                       header='y,outer,center,inner,width',
                       comments='')
            print(f"[export_csv] Saved {len(self.det.ys)} points → {path}")
            return

        df = self._to_dataframe()
        df.to_csv(path, index=False)
        print(f"[export_csv] Saved {len(df)} rows → {path}")

    def export_excel(self, path: str = "points.xlsx"):
        """Export detected points to Excel with a diff-array sheet."""

        df   = self._to_dataframe()
        arr  = self._get_arr()

        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # Sheet 1: detected points
            df.to_excel(writer, sheet_name='Points', index=False)

            # Sheet 2: diff array (sampled — full array may be huge)
            if arr is not None:
                # Downsample to max 200x200 for readability
                step = max(1, max(arr.shape) // 200)
                arr_small = arr[::step, ::step]
                pd.DataFrame(arr_small).to_excel(
                    writer, sheet_name='DiffArray',
                    header=False, index=False,
                )

        print(f"[export_excel] Saved → {path}")

    def show_all(self, row: int | None = None, col: int | None = None):
        """
        Run all visualizations in sequence.
        row / col default to the middle of the image if not given.
        """
        h, w = self.det.gray.shape
        row  = row if row is not None else h // 2
        col  = col if col is not None else w // 2

        self.print_summary()
        self.show_diff_array()
        self.show_row_profile(row)
        self.show_col_profile(col)
        self.show_points_scatter()
        self.show_width_histogram()

    def _get_arr(self):
        """Return the diff/binary array from the detector, whichever exists."""
        if hasattr(self.det, 'diff_arr') and self.det.diff_arr is not None:
            return self.det.diff_arr
        return None

    def _to_dataframe(self):
        import pandas as pd
        return pd.DataFrame({
            'y':      self.det.ys,
            'outer':  self.det.outers,
            'center': self.det.centers,
            'inner':  self.det.inners,
            'width':  self.det.widths,
        })

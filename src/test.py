import plot
from field_detection import FieldDetection
import subprocess
import numpy as np
import math
import cv2
import os

if __name__ == "__main__":

    cwd = os.getcwd()

    FRAME_NR = 60
    QUADRADO = 1
    WINDOW_NAME = "BOUNDARY DETECTION"
    VERTICAL_LINES_NR = 60

    # FIELD DETECTION TESTS
    field_detector = FieldDetection(
                    vertical_lines_offset=320,
                    vertical_lines_nr=12,
                    min_line_length=1,
                    max_line_length=20,
                    min_wall_length=10,
                    arrange_random=False)
    print(field_detector.vertical_lines)

    while True:
        IMG_PATH = cwd + f"/data/quadrado{QUADRADO}/{FRAME_NR}.jpg"
        img = cv2.imread(IMG_PATH)

        boundary_points, line_points = field_detector.detectFieldLinesAndBoundary(img)

        for point in boundary_points:
            pixel_y, pixel_x = point
            img[pixel_y, pixel_x] = field_detector.RED

        for point in line_points:
            pixel_y, pixel_x = point
            img[pixel_y, pixel_x] = field_detector.BLUE
            
  
        cv2.imshow(WINDOW_NAME, img)
        # plot.plot_boundary_orientation(field_detector.boundary)
        plot.plot_gaussian_distribution(field_detector.boundary)


        subprocess.call(['xdotool', 'search', '--name', WINDOW_NAME, 'windowactivate'])


        key = cv2.waitKey(-1) & 0xFF
        if key == ord('q'):
            plot.plt.close('all')
            break
        else:
            plot.plt.close('all')
            FRAME_NR += 1

    # RELEASE WINDOW AND DESTROY
    cv2.destroyAllWindows()
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
                    vertical_lines_nr=15,
                    min_line_length=1,
                    max_line_length=20,
                    min_wall_length=15,
                    arrange_random=False)

    while True:
        IMG_PATH = cwd + f"/data/pibic_evaluate/ig_02/201.png"

        img = cv2.imread(IMG_PATH)
 
        boundary_points, window_img = field_detector.detectFieldLinesAndBoundary(img)
        
        print(f'boundary_points = {boundary_points}')

        for p in boundary_points:
            pixel_y, pixel_x = p
            window_img[pixel_y, pixel_x] = field_detector.BLUE
            
        cv2.imshow(WINDOW_NAME, window_img)

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
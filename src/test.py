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
                    vertical_lines_nr=5,
                    min_line_length=1,
                    max_line_length=20,
                    min_wall_length=5,
                    arrange_random=False)
    print(field_detector.vertical_lines)

    while True:
        IMG_PATH = cwd + f"/data/quadrado{QUADRADO}/{FRAME_NR}.jpg"

        print(f'FRAME_NR = {FRAME_NR}')

        img = cv2.imread(IMG_PATH)
        first_lines_img = img.copy()

        preprocessed = field_detector.preprocess(first_lines_img, field_detector.vertical_lines)
        print_img = preprocessed.copy()
        segmented_img = field_detector.segmentField(preprocessed, field_detector.vertical_lines)
        boundary_points, orientation, sobel_img = field_detector.fieldWallDetection(segmented_img, img)


        field_detector.boundary = boundary_points

    

        for point in boundary_points:
            pixel_y, pixel_x = point
            sobel_img[pixel_y, pixel_x] = field_detector.BLUE
        
        cv2.imshow(WINDOW_NAME, sobel_img)

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
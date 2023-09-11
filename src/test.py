import plot
from field_detection import FieldDetection
import subprocess
import numpy as np
import math
import glob
import cv2
import os

if __name__ == "__main__":

    cwd = os.getcwd()

    FRAME_NR = 600
    QUADRADO = 1
    WINDOW_NAME = "BOUNDARY DETECTION"
    VERTICAL_LINES_NR = 60

    # FIELD DETECTION TESTS
    field_detector = FieldDetection(
                    vertical_lines_offset=320,
                    vertical_lines_nr=1,
                    min_line_length=1,
                    max_line_length=20,
                    min_wall_length=5,
                    arrange_random=False)
    print(field_detector.vertical_lines)


    # Specify the directory path for which you want to get subdirectories
    directory_path = 'data/pibic_evaluate'

    # Initialize an empty list to store the directory paths
    subdirectories = []

    # Use os.listdir() to get a list of all items (files and directories) in the specified directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        # Check if the item is a directory using os.path.isdir()
        if os.path.isdir(item_path):
            subdirectories.append(item_path)

    for directory_path in subdirectories:

        archive_extensions = ['.png'] 

        archive_paths = []

        for extension in archive_extensions:
            pattern = os.path.join(directory_path, f"*{extension}")
            archive_paths.extend(glob.glob(pattern))

        for path in archive_paths:

            IMG_PATH = cwd + f"/{path}"

            WINDOW_NAME = IMG_PATH

            print(f'IMG_PATH')
            img = cv2.imread(IMG_PATH)
            hough_image = img.copy()
            lr_image = img.copy()
    
            boundary_points_r, window_img_r = field_detector.processLinearRegression(hough_image)
            cv2.imshow(WINDOW_NAME, window_img_r)



            # boundary_points_h, window_img_h=field_detector.detectFieldLinesAndBoundary(lr_image)
            # cv2.imshow("Hough Image", window_img_h)
            

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
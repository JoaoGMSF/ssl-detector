from numba import cuda
import numpy as np
import time
import math
import cv2
import os

class FieldDetection():
    def __init__(
            self,
            vertical_lines_offset = 320,
            vertical_lines_nr = 1,
            min_line_length = 1,
            max_line_length = 20,
            min_wall_length = 10
            ):
        # DEFINE COLORS:
        self.BLACK = [0, 0, 0]
        self.BLUE = [255, 0, 0]
        self.GREEN = [0, 255, 0]
        self.RED = [0, 0, 255]
        self.WHITE = [255, 255, 255]

        # min/max amount of pixels for detection
        self.min_line_length = min_line_length
        self.max_line_length = max_line_length
        self.min_wall_length = min_wall_length

        # line scans offset
        self.vertical_lines = []
        self.vertical_lines_nr = vertical_lines_nr
        self.arrangeVerticalLinesUniform(vertical_lines_offset, img_width=640)

        self.mask_points = []
    
    def arrangeVerticalLinesUniform(self, vertical_lines_offset = 320, img_width = 640):
        vertical_lines = []
        for line_x in range(vertical_lines_offset, self.vertical_lines_nr*vertical_lines_offset+1, vertical_lines_offset):
            if line_x>5 and line_x<img_width-5: vertical_lines.append(line_x)
            else: print(f"Detection line out of resolution bounds! Vertical position:Line {line_x}")
        self.vertical_lines = vertical_lines

    def arrangeVerticalLinesRandom(self, img_width = 640):
        vertical_lines = []
        for i in range(self.vertical_lines_nr):
            line_x = int(np.random.uniform(5, img_width-5))
            vertical_lines.append(line_x)
        self.vertical_lines = vertical_lines

    def isBlack(self, src):
        blue, green, red = src
        if green < 70 and red < 70 and blue < 70:
            return True
        else:
            return False
    
    def isGreen(self, src):
        blue, green, red = src
        if green > 90 and red < 110:
            return True
        else:
            return False     

    def isWhite(self, src):
        blue, green, red = src
        if blue > 130 and green > 130 and red > 130:
            return True
        else:
            return False

    def segmentPixel(self, src):
        dis_black = abs(src[1]-0)
        dis_green = abs(src[1]-255)
        minimo = min(dis_black,dis_green)
        if minimo == dis_black:
            color = self.BLACK
            return color
        else:
            src_2d_vector = np.mean([src[0],src[2]])
            segmentation_treshold = 210
            if src_2d_vector>segmentation_treshold:
                color = self.WHITE
                return color
            elif src_2d_vector<=segmentation_treshold:
                color = self.GREEN
                return color           
            else:
                return src
    
    @cuda.jit
    def segmentField(src):
        """
        Make description here
        """
        x = cuda.threadIdx.x
        y = cuda.blockIdx.x 
        
        
        rmse_black = math.sqrt((((src[x,y,0])**2)+((src[x,y,2])**2))/2)
        if rmse_black <= 50:
            src[x,y] = [0,0,0]
        else:
            rmse_white = math.sqrt((((src[x,y,0]-255)**2)+((src[x,y,2]-255)**2))/2)
            segmentation_treshold = 210
            if rmse_white <= 40:
                src[x,y] = [255,255,255]
            else:
                src[x,y] = [0,255,0]

    def fieldWallDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # wall detection points
        boundary_points = []

        for line_x in self.vertical_lines:
            wall_points = []
            for pixel_y in range(height-1, 0, -1):
                pixel = src[pixel_y, line_x]
                if len(wall_points)>self.min_wall_length:
                    boundary_points.append(wall_points[0])
                    break
                elif self.isBlack(pixel):
                    wall_points.append([pixel_y, line_x])
                else:
                    wall_points = []

        return boundary_points

    def fieldLineDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # field lines detection points
        field_line_points = []

        for line_x in self.vertical_lines:
            field_line = False
            line_points = []
            for pixel_y in range(height-1, 0, -1):
                pixel = src[pixel_y, line_x]
                if not self.isBlack(pixel):
                    # check white->green border
                    if not self.isGreen(pixel) and field_line == False:
                        field_line = True
                    # check white->green border
                    elif self.isGreen(pixel) and field_line == True:
                        field_line = False
                        # check white line length (width)
                        if len(line_points)>self.min_line_length and len(line_points)<self.max_line_length:
                            line_y = [point[0] for point in line_points]
                            point = int(np.mean(line_y)), line_x
                            field_line_points.append(point)
                        line_points = []

                    if field_line == True:
                        line_points.append([pixel_y, line_x])

        return field_line_points      

    def detectFieldLinesAndBoundary(self, src):
        """
        Make description here
        """

        lines, _, _ = src.shape
        # threads_per_block = (int(lines / 4))
        # blocks_per_grid = (int(4 * len(self.vertical_lines)))

        threads_per_block = lines
        blocks_per_grid = len(self.vertical_lines)


        segmented_columns = np.ascontiguousarray(src[:,self.vertical_lines])
        
        start = time.time()
        
        self.segmentField[blocks_per_grid, threads_per_block](segmented_columns)
        
        end = time.time() - start

        for i,j in enumerate(self.vertical_lines):
            src[:,j] = segmented_columns[:,i]

        print(f"tempo de execução pra gerar linhas {end}")
        
        # segmented_img = self.segmentField(src)
        boundary_points = self.fieldWallDetection(src)
        field_line_points = self.fieldLineDetection(src)
        return boundary_points, field_line_points     

    def detectFieldLinesAndBoundaryMerged(self, src):
        """
        Make description here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # wall detection points
        boundary_points = []

        # field lines detection points
        field_line_points = []

        for line_x in self.vertical_lines:
            wall_points = []
            for pixel_y in range(height-1, 0, -1):
                pixel = src[pixel_y, line_x]
                if len(wall_points)>self.min_wall_length:
                    boundary_points.append(wall_points[0])
                    break
                elif self.isBlack(pixel):
                    wall_points.append([pixel_y, line_x])
                else:
                    wall_points = []
        
        return boundary_points, field_line_points       


if __name__ == "__main__":
    from glob import glob

    cwd = os.getcwd()

    FRAME_NR = 5
    QUADRADO = 1
    WINDOW_NAME = "BOUNDARY DETECTION"
    VERTICAL_LINES_NR = 1

    # FIELD DETECTION TESTS
    field_detector = FieldDetection(
                    vertical_lines_offset=320,
                    vertical_lines_nr=20,
                    min_line_length=1,
                    max_line_length=20,
                    min_wall_length=10)
    print(field_detector.vertical_lines)

    while True:
        IMG_PATH = cwd + f"/data/quadrado{QUADRADO}/{FRAME_NR}.jpg"
        file = glob(IMG_PATH)
        img = cv2.imread(IMG_PATH)

        field_detector.arrangeVerticalLinesRandom()

        start = time.time()

        boundary_points, line_points = field_detector.detectFieldLinesAndBoundary(img)

        for point in boundary_points:
            pixel_y, pixel_x = point
            img[pixel_y, pixel_x] = field_detector.RED

        for point in line_points:
            pixel_y, pixel_x = point
            img[pixel_y, pixel_x] = field_detector.RED
            
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(-1) & 0xFF
        if key == ord('q'):
            break
        else:
            FRAME_NR += 1

    # RELEASE WINDOW AND DESTROY
    cv2.destroyAllWindows()
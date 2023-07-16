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
            min_wall_length = 10,
            arrange_random = False
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
        self.arrangeLines(arrange_random)

        #points detectes
        self.boundary = []

        self.segmented_image=[[[]]]

        self.mask_points = []
    
    def arrangeLines(self, arrange_random):
        if arrange_random:
            self.arrangeVerticalLinesRandom(img_width=640)
        else:
            self.arrangeVerticalLinesUniform(img_width=640)

    def arrangeVerticalLinesUniform(self, img_width = 640):
        vertical_lines = []
        vertical_lines_offset = (img_width)/(self.vertical_lines_nr+1)
        for line_x in range(int(vertical_lines_offset), int(self.vertical_lines_nr*vertical_lines_offset)+1, int(vertical_lines_offset)):
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
        hue, saturation, value = src
        if (hue>=55 and hue<=85) and saturation>=50:
            src = self.GREEN
        else:
            if value>=100:
                src = self.WHITE
            else:
                src = self.BLACK    
        return src

    def validateBoundaryPoints(boundary, height_threshold=20):
        pass

    def projectBoundaryLine(self, y1, x1, orientation, is_degree = False):
        if is_degree: orientation = np.deg2rad(orientation)
        a = np.tan(orientation)
        b = (y1-1) - a*(x1-1)
        x2 = int(x1+40)
        y2 = int(a*x2 + b)
        return (x1, y1), (x2, y2)

    def getBoundaryPointsOrientation(self, src, boundary_points):
        boundary_ground_points_orientation = []
        for point in boundary_points:
            pixel_y, pixel_x = point
            # paint pixel for debug and documentation
            src[pixel_y, pixel_x] = self.RED
            cv2.drawMarker(src, (pixel_x, pixel_y), color=self.RED)
            _, orientation = self.sobel(src, point)
            boundary_ground_points_orientation.append(([pixel_y,pixel_x], orientation))
        return boundary_ground_points_orientation

    def sobel(self, src, pixel):
        pixel_y, pixel_x = pixel
        A = src[pixel_y-1:pixel_y+2, pixel_x-1:pixel_x+2]
        # blur = cv2.GaussianBlur(A,(3,3),0)
        gray = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
        kernel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        kernel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        #import pdb;pdb.set_trace()
        Gx = sum(sum(kernel_x*gray))
        Gy = sum(sum(kernel_y*gray))
        magnitude = np.sqrt(Gx**2 + Gy**2)
        orientation = np.arctan2(Gy, Gx)
        pt1, pt2 = self.projectBoundaryLine(pixel_y, pixel_x, orientation+np.pi/2)
        pt1_y= int(pt1[0])
        pt1_x= int(pt1[1])
        pt2_y= int(pt2[0])
        pt2_x= int(pt2[1])
        # import pdb;pdb.set_trace()
        cv2.line(src,pt1,pt2,color=(0,255,0), thickness=2)

        return magnitude, np.rad2deg(orientation)


    def segmentField(self, src):
        """
        Make description here
        """
        # make copy from source image for segmentation
        # segmented_img = src.copy()
        hsv = src
        segmented_img = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        for line_x in self.vertical_lines:
            # segment vertical lines
            for pixel_y in range(0, height):
                pixel = segmented_img[pixel_y, line_x]
                color = self.segmentPixel(pixel)
                segmented_img[pixel_y, line_x] = color

        return segmented_img        
    

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

        segmented_img = self.segmentField(src)
        boundary_points = self.fieldWallDetection(src)
        field_line_points = self.fieldLineDetection(src)
        self.boundary = self.getBoundaryPointsOrientation(src, boundary_points=boundary_points)

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
        #file = glob(IMG_PATH)
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
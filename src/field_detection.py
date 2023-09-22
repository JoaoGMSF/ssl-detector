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
            min_goal_length = 30,
            height_threshold = 10,
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
        self.min_goal_length = min_goal_length
        self.height_threshold = height_threshold

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
        hue, light, saturation = src
        if (hue>=47 and hue<=112) and saturation>60:
            src = [0,0,255]

        else:
            if light>=85:
                src = [255,255,255]
            else:
                src = [0,0,0]
        return src

    def segmentPixelWindow(self, src):
        hue, saturation, value = src
        if value > 90:
            src = [0,0,255]
        else:
            src = [0,0,0]
        return src
    
    def getBoundaryPointsOrientation(self, src, boundary_points):
        """
        Recebe os pontos de borda, aplica o sobel, printa uma cruz no ponto, e retorna os pontos de borda no formato
        ([pixel_y,pixel_x], orientation)
        """

        sobel_img = src
        orientation_list = []
        boundary_ground_points_orientation = []
        for point in boundary_points:
            sobel_img[point[0]-2:point[0]+3, point[1]-2:point[1]+3] = self.preprocessWindow(src, point)
            pixel_y, pixel_x = point
            # paint pixel for debug and documentation
            sobel_img[pixel_y, pixel_x] = self.RED
            cv2.drawMarker(src, (pixel_x, pixel_y), color=self.RED)
            _, orientation = self.sobel(src, point)
            boundary_ground_points_orientation.append(([pixel_y,pixel_x], orientation))
            orientation_list.append(orientation)

        return boundary_ground_points_orientation, sobel_img, orientation


    def sobel(self, src, pixel):
        pixel_y, pixel_x = pixel
        A = src[pixel_y-2:pixel_y+3, pixel_x-2:pixel_x+3]
        # blur = cv2.GaussianBlur(A,(3,3),0)
        gray = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
        kernel_x = np.array([
            [5,   4,  0,   -4,   -5],
            [8,  10,  0,  -10,   -8],
            [10, 20,  0,  -20,  -10],
            [8,  10,  0,  -10,   -8],
            [5,   4,  0,   -4,   -5]
        ])

        kernel_y = np.array([
            [-5,  -8, -10,  -8, -5],
            [-4, -10, -20, -10, -4],
            [ 0,   0,   0,   0,  0],
            [ 4,  10,  20,  10,  4],
            [ 5,   8,  10,   8,  5],
        ])

        Gx = sum(sum(kernel_x*gray))
        Gy = sum(sum(kernel_y*gray))
        magnitude = np.sqrt(Gx**2 + Gy**2)
        orientation = np.arctan2(Gy, Gx)
        pt1, pt2 = self.projectBoundaryLine(pixel_y, pixel_x, orientation, 50)

        cv2.line(src,pt1,pt2,color=(0,255,0), thickness=2)

        return magnitude, np.rad2deg(orientation)

    def preprocess(self, img, lines):
        splited_img = None
        new_img = img
        for line in lines:
            splited_img = img[:, line-2:line+3]
            kernel = np.ones((5, 5), np.uint8)
            img_dilation = cv2.erode(splited_img, kernel, iterations=1)
            blur = cv2.GaussianBlur(img_dilation,(3,3),0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
            new_img[:, line-2:line+3] = hsv 
        
        return new_img
    
    def preprocessWindow(self, img, point, boundary_window=2):
        splited_img = img[point[0]-boundary_window:point[0]+boundary_window+1, point[1]-boundary_window:point[1]+boundary_window+1]
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.erode(splited_img, kernel, iterations=1)
        blur = cv2.GaussianBlur(img_dilation,(3,3),0)

        preprocessed_window = blur

        return preprocessed_window

    def projectBoundaryLine(self, y1, x1, orientation, line_length, is_degree = False):
        """
        função auxiliar para printar uma linha na imagem
        """
        if is_degree: orientation = np.deg2rad(orientation)
        x2 = int(x1 + line_length * math.cos(orientation))
        y2 = int(y1 - line_length * math.sin(orientation))

        return (x1, y1), (x2, y2)



    def segmentField(self, src, lines):
        """
        Make description here
        """
        # make copy from source image for segmentation
        # segmented_img = src.copy()
        segmented_img = src

        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        for line_x in lines:
            # segment vertical lines
            for pixel_y in range(0, height):
                pixel = segmented_img[pixel_y, line_x]
                color = self.segmentPixel(pixel)
                segmented_img[pixel_y, line_x] = color

        return segmented_img        
    

    def segmentWindow(self, src):
        """
        Make description here
        """
        # make copy from source image for segmentation
        # segmented_img = src.copy()
        segmented_img = src

        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        for line_x in range(0, width):
            # segment vertical lines
            for pixel_y in range(0, height):
                pixel = segmented_img[pixel_y, line_x]
                color = self.segmentPixelWindow(pixel)
                segmented_img[pixel_y, line_x] = color

        return segmented_img  

    def findBoundaryPoints(self, src, lines):
        height, width = src.shape[0], src.shape[1]
        boundary_points = []

        for line_x in lines:
            wall_points = []
            for pixel_y in range(height-1, 0, -1):
                pixel = src[pixel_y, line_x]
                if len(wall_points)>self.min_wall_length:
                    false_boundary = 0
                    for i in range(10):
                        if self.isWhite(src[wall_points[0][0]+i,wall_points[0][1]]):
                            false_boundary += 1
                    if false_boundary > 7:
                        pass
                    else:
                        boundary_points.append(wall_points[0])
                    break
                elif self.isBlack(pixel):
                    wall_points.append([pixel_y, line_x])
                else:
                    wall_points = []

        return boundary_points

    def findBoundaryWindow(self, window_img, boundary_points, boundary_threshold):
        window_boundary_points = []

        segmented_img = window_img.copy()

        for point in boundary_points:
            # boundary_window = src[point[0]-BOUNDARY_WINDOW:point[0]+BOUNDARY_WINDOW, point[1]-BOUNDARY_WINDOW:point[1]+BOUNDARY_WINDOW]
            boundary_window = self.preprocessWindow(window_img, point, boundary_threshold)

            padding_y = point[0]-boundary_threshold
            padding_X = point[1]-boundary_threshold
            segmented_window = self.segmentWindow(boundary_window)

            segmented_img[point[0]-boundary_threshold:point[0]+boundary_threshold, point[1]-boundary_threshold:point[1]+boundary_threshold] = segmented_window

            segmented_height, segmented_width, _ = segmented_window.shape

            for line_x in range(0,segmented_width,5):
                wall_points = []
                for pixel_y in range(segmented_height-1, 0, -1):
                    pixel = segmented_window[pixel_y, line_x]
                    if len(wall_points)>self.min_wall_length:
                        window_boundary_points.append(wall_points[0])
                        break
                    elif self.isBlack(pixel):
                        wall_points.append([pixel_y+padding_y, line_x+padding_X])
                    else:
                        wall_points = []

        return window_boundary_points, segmented_img

    def fieldWallDetection(self, boundary_img, sobel_img):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = boundary_img.shape[0], boundary_img.shape[1]
        BOUNDARY_WINDOW = 20

        # wall detection points
        boundary_points = []

        #find boundaries in find_boundary_img
        boundary_points = self.findBoundaryPoints(boundary_img, self.vertical_lines)
        
        boundary_orientation, sobel_img, orientation = self.getBoundaryPointsOrientation(sobel_img, boundary_points)

        # ymean_boundary_points = np.mean([point[0] for point in boundary_points])
        # xmean_boundary_points = np.mean([point[1] for point in boundary_points])

        

        return boundary_points, orientation, sobel_img

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
        self.boundary  = boundary_points

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
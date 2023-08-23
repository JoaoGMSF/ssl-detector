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
        hue, saturation, value = src
        if (hue>=55 and hue<=85) and saturation>=70:
            src = [60,255,255]
        else:
            if value>=100:
                src = [0,0,255]
            else:
                src = [0,0,0]
        return src

    def preprocess(self, img, lines):
        splited_img = None
        new_img = img
        for line in lines:
            splited_img = img[:, line-2:line+3]
            kernel = np.ones((5, 5), np.uint8)
            img_dilation = cv2.erode(splited_img, kernel, iterations=1)
            blur = cv2.GaussianBlur(img_dilation,(3,3),0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            new_img[:, line-2:line+3] = hsv 
        
        return new_img
    
    def preprocessWindow(self, img, point, boundary_window):
        splited_img = img[point[0]-boundary_window:point[0]+boundary_window, point[1]-boundary_window:point[1]+boundary_window]
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.erode(splited_img, kernel, iterations=1)
        blur = cv2.GaussianBlur(img_dilation,(3,3),0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        cv2.imshow("hsv", hsv)

        key = cv2.waitKey(-1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
        return hsv

    def projectBoundaryLine(self, y1, x1, orientation, is_degree = False):
        """
        função auxiliar para printar uma linha na imagem
        """
        if is_degree: orientation = np.deg2rad(orientation)
        a = np.tan(orientation)
        b = (y1-1) - a*(x1-1)
        x2 = int(x1+40)
        y2 = int(a*x2 + b)
        return (x1, y1), (x2, y2)

    def getBoundaryPointsOrientation(self, src, boundary_points):
        """
        Recebe os pontos de borda, aplica o sobel, printa uma cruz no ponto, e retorna os pontos de borda no formato
        ([pixel_y,pixel_x], orientation)
        """
        boundary_ground_points_orientation = []
        for point in boundary_points:
            pixel_y, pixel_x = point
            # paint pixel for debug and documentation
            src[pixel_y, pixel_x] = self.RED
            cv2.drawMarker(src, (pixel_x, pixel_y), color=self.RED)
            _, orientation = self.sobel(src, point)
            boundary_ground_points_orientation.append(([pixel_y,pixel_x], orientation))
        return boundary_ground_points_orientation


    def line_detection_non_vectorized(self, image,  boundary_points, num_rhos=1000, num_thetas=1000, t_count=220):
        
        boundary = boundary_points
        edge_height, edge_width, _ = image.shape
        #
        d = np.sqrt(np.square(edge_height) + np.square(edge_width))
        dtheta = 180 / num_thetas
        drho = (2 * d) / num_rhos
        #
        thetas = np.arange(0, 180, step=dtheta)
        rhos = np.arange(-d, d, step=drho)

        #
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))

        #
        accumulator = np.zeros((len(rhos), len(rhos)))
        #

        for edge_point in boundary:
            for theta_idx in range(len(thetas)):
                rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
                theta = thetas[theta_idx]
                rho_idx = np.argmin(np.abs(rhos - rho))
                accumulator[rho_idx][theta_idx] += 1

        maximun_acc = []

        THRESHOLD_ACC = 5

        for r in range(accumulator.shape[0]):
            for t in range(accumulator.shape[1]):
                if accumulator[r][t] > 5:
                    if(len(maximun_acc)==0):
                        maximun_acc.append([r,t])
                    else:
                        THRESHOLD_THETA = 50
                        isClose=False
                        isMaximun=False
                        for i in range(len(maximun_acc)):
                            #se tiver perto
                            if(np.abs(maximun_acc[i][1]-t)<=THRESHOLD_THETA):
                                #se o acc source tiver perto do acc dest, e o acc source for maior que o dest
                                if(accumulator[maximun_acc[i][0]][maximun_acc[i][1]]<=accumulator[r][t] and (not isClose)):
                                    isMaximun=True
                                    maximun_acc[i] = [r,t]
                                isClose=True
                            #se tiver distante
                            else:
                                pass
                        if((not isClose) and (not isMaximun)):
                            maximun_acc.append([r,t])
                            

        NUM_RETAS = 2    

        # indices_maior = np.unravel_index(np.argpartition(-accumulator, NUM_RETAS, axis=None)[:NUM_RETAS], accumulator.shape)

        def draw_hough_lines(img, rho, theta, color=[0, 255, 0], thickness=2):
            a = np.cos(np.deg2rad(theta))
            b = np.sin(np.deg2rad(theta))
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)   
                    
        print(f'maximun_acc = {maximun_acc}')

        # Imprima os índices e os valores dos três maiores elementos
        for i in maximun_acc:
            row, col = i[0], i[1]
            valor = accumulator[row, col]
            draw_hough_lines(image,rhos[row], thetas[col])
            print(f"Índice: ({row}, {col}), Valor: {valor}")
        return accumulator, rhos, thetas


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
                color = self.segmentPixel(pixel)
                segmented_img[pixel_y, line_x] = color

        return segmented_img        

    def fieldWallDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]
        BOUNDARY_WINDOW = 20

        # wall detection points
        boundary_points = []
        window_boundary_points = []

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


        for point in boundary_points:

            # boundary_window = src[point[0]-BOUNDARY_WINDOW:point[0]+BOUNDARY_WINDOW, point[1]-BOUNDARY_WINDOW:point[1]+BOUNDARY_WINDOW]
            boundary_window = self.preprocessWindow(src, point, BOUNDARY_WINDOW)

            padding_y = point[0]-BOUNDARY_WINDOW
            padding_X = point[1]-BOUNDARY_WINDOW
            segmented_window = self.segmentWindow(boundary_window)

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

        

        return boundary_points, window_boundary_points

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
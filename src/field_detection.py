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
        if light>=60:
            src = [0,0,255]
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
    
    def preprocessWindow(self, img, point, boundary_window):
        splited_img = img[point[0]-boundary_window:point[0]+boundary_window, point[1]-boundary_window:point[1]+boundary_window]
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.erode(splited_img, kernel, iterations=1)
        blur = cv2.GaussianBlur(img_dilation,(5,5),0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        preprocessed_window = hsv

        return preprocessed_window

    def line_detection_non_vectorized(self, image,  boundary_points, num_rhos=480, num_thetas=180, t_count=220):
        
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
        accumulator = np.zeros((len(rhos), len(thetas)))
        #
        print(f"boundary_points = {boundary}")
        for edge_point in boundary:
            for theta_idx in range(len(thetas)):
                rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
                theta = thetas[theta_idx]
                rho_idx = np.argmin(np.abs(rhos - rho))
                accumulator[rho_idx][theta_idx] += 1
                if accumulator[rho_idx][theta_idx] == 8:
                    print(f"rho = {rhos[rho_idx]}, theta = {thetas[theta_idx]}")


        #TRATAMENTO DE VÁRIAS LINHAS

        maximun_acc = []
        average_parameters = []

        THRESHOLD_ACC = 5

        for r in range(accumulator.shape[0]):
            for t in range(accumulator.shape[1]):
                if accumulator[r][t] > 3:
                    if(len(maximun_acc)==0):
                        maximun_acc.append([r,t])
                        average_parameters.append([(r,t)])
                    else:
                        THRESHOLD_THETA = 10
                        isClose=False
                        isMaximun=False
                        for i in range(len(maximun_acc)):
                            #se tiver perto
                            if(np.abs(maximun_acc[i][1]-t)<=THRESHOLD_THETA):
                                #se o acc source tiver perto do acc dest, e o acc source for maior que o dest
                                if(accumulator[maximun_acc[i][0]][maximun_acc[i][1]]<=accumulator[r][t] and (not isClose)):
                                    print(f"acc of maximun = {accumulator[maximun_acc[i][0]][maximun_acc[i][1]]}, acc newly = {accumulator[r][t]}")
                                    if(accumulator[maximun_acc[i][0]][maximun_acc[i][1]]==accumulator[r][t]):
                                        average_parameters[i].append((r,t))
                                        print(f"average parameter added = {(r,t)} with acc_value = {accumulator[r][t]}")
                                    else:
                                        average_parameters[i] = []
                                        average_parameters[i].append((r,t))
                                        print("average_parameter empty")
                                        print(f"average parameter added = {(r,t)} with acc_value = {accumulator[r][t]}")
                                    isMaximun=True
                                    maximun_acc[i] = [r,t]
                                isClose=True
                            #se tiver distante
                            else:
                                pass
                        if((not isClose) and (not isMaximun)):
                            maximun_acc.append([r,t])
                            average_parameters.append([(r,t)])
                            
        print(f'average_parameters = {average_parameters}')

        maximun_average_acc = []

        for i in range(len(average_parameters)):
            average_rho = 0
            average_theta = 0
            for j in range(len(average_parameters[i])):
                average_rho += rhos[average_parameters[i][j][0]]
                average_theta += thetas[average_parameters[i][j][1]]
            maximun_average_acc.append([int(np.round(average_rho/len(average_parameters[i]))),int(np.round(average_theta/len(average_parameters[i])))])


        NUM_RETAS = 2    

        indices_maior = np.unravel_index(np.argpartition(-accumulator, NUM_RETAS, axis=None)[:NUM_RETAS], accumulator.shape)

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
        
        for parameters in maximun_average_acc:
            draw_hough_lines(image,parameters[0], parameters[1])
            print(f'Rho = {parameters[0]}, Theta = {parameters[1]}')
            print(f'maximun_acc = [{rhos[maximun_acc[0][0]]},{thetas[maximun_acc[0][1]]}]')
            print(f'maximun_average_acc = {maximun_average_acc}\n')

        return accumulator, rhos, thetas, maximun_average_acc

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

    def fieldWallDetection(self, boundary_img, window_img):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = boundary_img.shape[0], boundary_img.shape[1]
        BOUNDARY_WINDOW = 20

        # wall detection points
        boundary_points = []
        window_boundary_points = []

        #find boundaries in find_boundary_img
        boundary_points = self.findBoundaryPoints(boundary_img, self.vertical_lines)

        #find boundary points in window
        # window_boundary_points, segmented_img = self.findBoundaryWindow(window_img, boundary_points, BOUNDARY_WINDOW)
        
        # return boundary_points, window_boundary_points, segmented_img
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
        first_line_img = src.copy()
        preprocessed = self.preprocess(first_line_img, self.vertical_lines)
        segmented_img = self.segmentField(preprocessed, self.vertical_lines)
        # boundary_points, window_boundary_points, window_img = self.fieldWallDetection(segmented_img, src)
        boundary_points = self.fieldWallDetection(segmented_img, src)

        self.boundary  = boundary_points

        acc, rhos, theta, parameters = self.line_detection_non_vectorized(image=segmented_img, boundary_points=boundary_points)

        rmse = self.validate(boundary_points, parameters)

        print(f'rmse = {rmse}')

        return boundary_points, first_line_img 

    def validate(self, boundary_points, parameters):
        
        sum_difs = 0

        for point in boundary_points:
            y_line = 0
            for param in parameters:
                rho = param[0]
                theta = param[1]
                y_line_aux = (rho - (point[1]*np.cos(np.deg2rad(theta))))/np.sin(np.deg2rad(theta))
                if y_line_aux>y_line:
                    y_line = y_line_aux

            y_point = point[0]
            sum_difs += (y_line-y_point)**2
            print(f"y_line = {y_line}; y_point = {y_point}")

        
        rmse = math.sqrt(sum_difs/len(boundary_points))

        return rmse

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
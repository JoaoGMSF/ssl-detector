from entities import Ball, Robot, Goal, Field, Frame
from field_detection import FieldDetection
from object_localization import Camera
try:
    from object_detection import DetectNet
except:
    print("Could not import tensorrt and its dependencies")
import numpy as np
import cv2
import os
import time

class JetsonVision():
    cwd = os.getcwd()

    # OBJECT DETECTION MODEL
    PATH_TO_MODEL = cwd+"/models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt"
    PATH_TO_LABELS = cwd+"/models/ssl_labels.txt"

    # CAMERA PARAMETERS SETUP
    PATH_TO_INTRINSIC_PARAMETERS = cwd+"/configs/mtx.txt"
    PATH_TO_2D_POINTS = cwd+"/configs/calibration_points2d.txt"
    PATH_TO_3D_POINTS = cwd+"/configs/calibration_points3d.txt"
    camera_matrix = np.loadtxt(PATH_TO_INTRINSIC_PARAMETERS, dtype="float64")
    calibration_position = np.loadtxt(cwd+"/configs/camera_initial_position.txt", dtype="float64")

    def __init__(
        self,
        vertical_lines_offset = 320,
        vertical_lines_nr = 1,
        model_path=PATH_TO_MODEL, 
        labels_path=PATH_TO_LABELS, 
        score_threshold = 0.5,
        draw = False,
        camera_matrix = camera_matrix,
        camera_initial_position=calibration_position,
        points2d_path = PATH_TO_2D_POINTS,
        points3d_path = PATH_TO_3D_POINTS,
        debug = False,
        enable_field_detection = True,
        enable_randomized_observations = False,
        min_wall_length = 10   
        ):

        try:
            self.object_detector = DetectNet(
                model_path=model_path,
                labels_path=labels_path,
                score_threshold=score_threshold,
                draw=draw
                )
            self.object_detector.loadModel()
            self.has_object_detection = True
        except:
            print("TensorRT not available, not running object detection!")
            self.has_object_detection = False
        
        self.jetson_cam = Camera(
            camera_matrix=camera_matrix,
            camera_initial_position=camera_initial_position
            )
        points2d = np.loadtxt(points2d_path, dtype="float64")
        points3d = np.loadtxt(points3d_path, dtype="float64")
        self.jetson_cam.computePoseFromPoints(points3d=points3d, points2d=points2d)
        
        self.field_detector = FieldDetection(
            vertical_lines_offset = vertical_lines_offset,
            vertical_lines_nr = vertical_lines_nr,
            min_wall_length = min_wall_length
            )
        self.enable_randomized_observations = enable_randomized_observations
        
        self.field = Field()
        self.current_frame = Frame()
        self.tracked_ball = Ball()
        self.tracked_goal = Goal()
        self.tracked_robot = Robot()

        self.debug_mode = debug
        self.has_field_detection = enable_field_detection

    def trackBall(self, score, xmin, xmax, ymin, ymax):
        # UPDATES BALL BASED ON DETECTION SCORE
        pixel_x, pixel_y = self.jetson_cam.ballAsPoint(
                                        left=xmin, 
                                        top=ymin, 
                                        right=xmax, 
                                        bottom=ymax)        
        x, y, w = self.jetson_cam.pixelToRobotCoordinates(pixel_x=pixel_x, pixel_y=pixel_y, z_world=0)
        ball = self.current_frame.updateBall(x, y, score)
        return ball
    
    def trackGoal(self, score, xmin, xmax, ymin, ymax):
        # UPDATES GOAL BASED ON DETECTION SCORE
        pixel_x, pixel_y = self.jetson_cam.goalAsPoint(
                                        left=xmin, 
                                        top=ymin, 
                                        right=xmax, 
                                        bottom=ymax)        
        x, y, w = self.jetson_cam.pixelToRobotCoordinates(pixel_x=pixel_x, pixel_y=pixel_y, z_world=0)
        goal = self.current_frame.updateGoalCenter(x, y, score)
        return goal

    def trackRobot(self, score, xmin, xmax, ymin, ymax):
        # UPDATES ROBOT BASED ON DETECTION SCORE
        pixel_x, pixel_y = self.jetson_cam.robotAsPoint(
                                        left=xmin, 
                                        top=ymin, 
                                        right=xmax, 
                                        bottom=ymax)        
        x, y, w = self.jetson_cam.pixelToRobotCoordinates(pixel_x=pixel_x, pixel_y=pixel_y, z_world=0)
        robot = self.current_frame.updateRobot(x, y, score)
        return robot

    def updateObjectTracking(self, detection):
        """
        Detection ID's:
        0: background
        1: ball
        2: goal
        3: robot

        Labels are available at: ssl-detector/models/ssl_labels.txt
        """
        class_id, score, xmin, xmax, ymin, ymax = detection
        if class_id == 1:
            ball = self.trackBall(score, xmin, xmax, ymin, ymax)
            self.tracked_ball = ball
        elif class_id == 2:
            goal = self.trackGoal(score, xmin, xmax, ymin, ymax)
            self.tracked_goal = goal
        elif class_id == 3:
            robot = self.trackRobot(score, xmin, xmax, ymin, ymax)
            self.tracked_robot = robot
        
    def detectAndTrackObjects(self, src):
        detections = self.object_detector.inference(src).detections
        
        for detection in detections:
            self.updateObjectTracking(detection)

    def sobel(self, src, pixel):
        pixel_y, pixel_x = pixel
        A = src[pixel_y-1:pixel_y+2, pixel_x-1:pixel_x+2]
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
        pt1, pt2 = self.project_boundary_line(pixel_y, pixel_x, orientation+np.pi/2)
        #print(f'pt1: {pt1}, pt2: {pt2}')
        cv2.line(src, pt1, pt2, color=(0,255,0), thickness=2)

        return magnitude, np.rad2deg(orientation)

    def project_boundary_line(self, y1, x1, orientation, is_degree = False):
        #import pdb;pdb.set_trace()
        if is_degree: orientation = np.deg2rad(orientation)
        a = np.tan(orientation)
        b = (y1-1) - a*(x1-1)
        x2 = int(x1+50)
        y2 = int(a*x2 + b)
        return (x1, y1), (x2, y2)


    def detectAndTrackFieldPoints(self, src):
        if self.has_object_detection: # if running on jetson, use optimized version
            boundary_points, line_points = self.field_detector.detectFieldLinesAndBoundaryMerged(src)
        else:
            boundary_points, line_points = self.field_detector.detectFieldLinesAndBoundary(src)
        
        boundary_ground_points = []
        for point in boundary_points:
            pixel_y, pixel_x = point
            # paint pixel for debug and documentation
            if self.debug_mode:
                src[pixel_y, pixel_x] = self.field_detector.RED
                cv2.drawMarker(src, (pixel_x, pixel_y), color=self.field_detector.RED)
                print(f"sobel: {self.sobel(src, point)}")
            x, y, w = self.jetson_cam.pixelToRobotCoordinates(pixel_x=pixel_x, pixel_y=pixel_y, z_world=0)
            boundary_ground_points.append([x, y, w])
        
        line_ground_points = []
        for point in line_points:
            pixel_y, pixel_x = point
            # paint pixel for debug and documentation
            # if self.debug_mode:
            if False:
                src[pixel_y, pixel_x] = self.field_detector.RED
            x, y, w = self.jetson_cam.pixelToRobotCoordinates(pixel_x=pixel_x, pixel_y=pixel_y, z_world=0)
            line_ground_points.append([x, y, w])
        
        return boundary_ground_points, line_ground_points

    def process(self, src, timestamp):
        """
        Detects and tracks objects, field lines and boundary points

        Params:
        src: image source (camera frame)
        timestamp: current timestamp
        -----------------
        Returns:
        current_frame: current frame containing flags to check for objects' detection
        tracked_ball: ball position from tracking
        tracked_goal: goal center position from tracking
        tracked_robot: robot position from tracking
        particle_filter_observations: observations used for self-localization algorithm
        """
        self.current_frame = Frame(timestamp=timestamp, input_source=src)
        if self.has_object_detection:
            self.detectAndTrackObjects(self.current_frame.input) # 30ms
        # 42ms with field lines detection, 8~9ms without it
        if self.has_field_detection:
            if self.enable_randomized_observations: self.field_detector.arrangeVerticalLinesRandom()
            particle_filter_observations = self.detectAndTrackFieldPoints(self.current_frame.input)
        else:
            particle_filter_observations = []
        processed_vision = self.current_frame, self.tracked_ball, self.tracked_goal, self.tracked_robot, particle_filter_observations
        # processed_vision = self.current_frame, self.tracked_ball, self.tracked_goal, self.tracked_robot

        return processed_vision
        
if __name__ == "__main__":
    import time
    from glob import glob

    cwd = os.getcwd()

    frame_nr = 324
    quadrado_nr = 1

    vision = JetsonVision(
                        vertical_lines_offset=320,
                        enable_randomized_observations=True,
                        debug=True)

    while True:
        dir = cwd + f"/data/quadrado{quadrado_nr}/{frame_nr}.jpg"

        WINDOW_NAME = "BOUNDARY DETECTION"
        img = cv2.imread(dir)
        height, width = img.shape[0], img.shape[1]
        _, _, _, _, particle_filter_observations = vision.process(img, timestamp=time.time())
        boundary_ground_points, line_ground_points = particle_filter_observations
        for point in boundary_ground_points:
            point = vision.jetson_cam.xyToPolarCoordinates(point[0], point[1])
            print(point)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(-1) & 0xFF
        if key == ord('q'):
            break
        else:
            frame_nr=frame_nr+1
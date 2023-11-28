###########################################
#       MAP AND THYMIO UTILITIES
###########################################
###########################################

import cv2
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import matplotlib.ticker as ticker

class EnvTracker:
    def __init__(self, map_size: tuple, cam_mtx: cv2.Mat, dist_coefs: cv2.Mat):
        """
        Init the environment tracker object to detect the map, the goal & the thymio.
        map_size: Tuple describing the width and height of the real map
        cam_mtx, dist_coeffs: Camera calibration parameters
        """
        # Define helpers & constants
        self.MARKER_DICT = cv2.aruco.DICT_4X4_50
        self.MAP_MARKER_IDS = (0,1,2,3)  # Indicate corner marker ids in order (BL, TL, TR, BR) (clockwise)
        self.MAP_MARKER_SIZE = 5        # In cm and should be square
        self.THYMIO_MARKER_IDS = (4,5)  # Markers on thymio (LEFT, RIGHT)
        self.MAP_MARKER_CORNERS = cv2.Mat(np.array([[0,0,0],
                                                  [0,self.MAP_MARKER_SIZE,0],
                                                  [self.MAP_MARKER_SIZE,self.MAP_MARKER_SIZE,0],
                                                  [self.MAP_MARKER_SIZE,0,0],
                                                  [0, map_size[1], 0],
                                                  [self.MAP_MARKER_SIZE, map_size[1], 0],
                                                  [self.MAP_MARKER_SIZE, map_size[1] - self.MAP_MARKER_SIZE, 0],
                                                  [0, map_size[1] - self.MAP_MARKER_SIZE, 0],
                                                  [map_size[0], map_size[1], 0],
                                                  [map_size[0], map_size[1] - self.MAP_MARKER_SIZE, 0],
                                                  [map_size[0] - self.MAP_MARKER_SIZE, map_size[1] - self.MAP_MARKER_SIZE, 0],
                                                  [map_size[0] - self.MAP_MARKER_SIZE, map_size[1], 0],
                                                  [map_size[0], 0, 0],
                                                  [map_size[0] - self.MAP_MARKER_SIZE, 0, 0],
                                                  [map_size[0] - self.MAP_MARKER_SIZE, self.MAP_MARKER_SIZE, 0],
                                                  [map_size[0], self.MAP_MARKER_SIZE, 0]], dtype=float))
        self.THYMIO_MARKER_SIZE = 3     # In cm
        self.THYMIO_MARKER_DIST = 10    # In cm
        # Define the corners of the 2 markers in "object" world (to have the center of the thymio as the axis frame reference)
        self.THYMIO_POSE_CORNERS = cv2.Mat(np.array([[-self.THYMIO_MARKER_SIZE/2, self.THYMIO_MARKER_SIZE/2, 0], 
                                                     [self.THYMIO_MARKER_SIZE/2, self.THYMIO_MARKER_SIZE/2, 0], 
                                                     [self.THYMIO_MARKER_SIZE/2, -self.THYMIO_MARKER_SIZE/2, 0], 
                                                     [-self.THYMIO_MARKER_SIZE/2, -self.THYMIO_MARKER_SIZE/2, 0]]))
        self.GOAL_MARKER_ID = 6         # Marker id for the goal location
        self.GOAL_MARKER_SIZE = 5       # In cm
        self.GOAL_POSE_CORNERS = cv2.Mat(np.array([[-self.GOAL_MARKER_SIZE/2, self.GOAL_MARKER_SIZE/2, 0], 
                                                [self.GOAL_MARKER_SIZE/2, self.GOAL_MARKER_SIZE/2, 0], 
                                                [self.GOAL_MARKER_SIZE/2, -self.GOAL_MARKER_SIZE/2, 0], 
                                                [-self.GOAL_MARKER_SIZE/2, -self.GOAL_MARKER_SIZE/2, 0]]))
        # Environment customs
        self.DETECTED_MAP_COLOR = [255,0,0]     # Blue (Don't forget OpenCV is BGR...)
        self.DETECTED_THYMIO_COLOR = [0,255,0]  # Green
        self.DETECTED_GOAL_COLOR = [0,0,255]    # Red

        # Saving arguments
        self._map_size = map_size
        self._cam_mtx = cam_mtx
        self._dist_coefs = dist_coefs

        # Map creation
        self.THRESHOLD = 100

        self._detected_markers = dict()
        self._map_corner_axes = tuple()
        self._thymio_axes = list()
        self._map_detected = False
        self._thymio_detected = False
        self._goal_detected = False
        self._goal_pose = np.zeros(2)
        self._gridmap = np.zeros((10,10)) # Arbitrary size

    def detectMarkers(self, frame: cv2.Mat) -> tuple[bool,cv2.Mat]:
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(self.MARKER_DICT))
        img_detect = cv2.copyTo(frame, None)
        corners, ids, rejected = detector.detectMarkers(img_detect)
        img_detect = cv2.aruco.drawDetectedMarkers(img_detect, corners, ids)
        # Save markers & ids
        if ids is not None:
            self._detected_markers = dict(zip(ids[:,0], corners))
        return ids is not None, img_detect
    
    def detectMap(self, frame: cv2.Mat) -> tuple[bool, cv2.Mat]:
        img_detect = cv2.copyTo(frame, None)
        if len(self._detected_markers.items()) == 0:
            self.detectMarkers(frame)
        self._map_detected = np.all(np.isin(list(self.MAP_MARKER_IDS), list(self._detected_markers.keys())))
        if self._map_detected :
            # Add a green rectangle to show detection
            for i in range(4):
                id = self.MAP_MARKER_IDS[i]
                next_id = self.MAP_MARKER_IDS[(i+1)%4]
                cv2.line(img_detect, self._detected_markers[id][0][0].astype(int), 
                         self._detected_markers[next_id][0][0].astype(int), self.DETECTED_MAP_COLOR, 2)
        return self._map_detected, img_detect

    def detectGoal(self, frame: cv2.Mat) -> tuple[bool, cv2.Mat]:
        img_detect = cv2.copyTo(frame, None)
        self._goal_detected = self.GOAL_MARKER_ID in self._detected_markers.keys()
        if (self._goal_detected):
            # Draw rectangle around goal
            corners = self._detected_markers[self.GOAL_MARKER_ID]
            for i in range(4):
                cv2.line(img_detect, corners[0][i].astype(int), corners[0][(i+1)%4].astype(int), self.DETECTED_GOAL_COLOR, 2)
        return self._goal_detected, img_detect
    
    def detectThymio(self, frame: cv2.Mat) -> tuple[bool, cv2.Mat]:
        img_detect = cv2.copyTo(frame, None)
        # Check if thymio markers were found in image
        self._thymio_detected = np.all(np.isin(self.THYMIO_MARKER_IDS, list(self._detected_markers.keys())))
        new_thymio_corners = np.zeros((4,2))
        if self._thymio_detected:
            # Left Marker
            new_thymio_corners[0,:] = self._detected_markers[self.THYMIO_MARKER_IDS[0]][0][0,:]
            new_thymio_corners[3,:] = self._detected_markers[self.THYMIO_MARKER_IDS[0]][0][3,:]
            # Right marker
            new_thymio_corners[1,:] = self._detected_markers[self.THYMIO_MARKER_IDS[1]][0][1,:]
            new_thymio_corners[2,:] = self._detected_markers[self.THYMIO_MARKER_IDS[1]][0][2,:]
            # Draw rectangle around markers
            for i in range(4):
                cv2.line(img_detect, new_thymio_corners[i].astype(int), new_thymio_corners[(i+1)%4].astype(int), self.DETECTED_THYMIO_COLOR, 2)
            # Extract main axis
            self._thymio_axes = []
            sign = 1
            for i in range(2):
                current_id = self.THYMIO_MARKER_IDS[i]
                next_id = self.THYMIO_MARKER_IDS[(i+1) % 2]
                corners = np.concatenate((self._detected_markers[current_id][0], self._detected_markers[next_id][0]))
                obj_corners = self.THYMIO_POSE_CORNERS.copy()
                obj_corners[:,0] = obj_corners[:,0] + sign*self.THYMIO_MARKER_DIST
                obj_corners = cv2.Mat(np.concatenate((self.THYMIO_POSE_CORNERS[:], obj_corners[:])))
                ret, rvec_thymio, tvec_thymio = cv2.solvePnP(obj_corners, corners, self._cam_mtx, self._dist_coefs, cv2.SOLVEPNP_EPNP)
                self._thymio_axes.append((rvec_thymio, tvec_thymio))
                img_detect = cv2.drawFrameAxes(img_detect, self._cam_mtx, self._dist_coefs, rvec_thymio, tvec_thymio, 2, 4)
                sign = -1*sign
            
        return self._thymio_detected, img_detect
        
    def extractMapCornerPose(self, frame: cv2.Mat) -> cv2.Mat:
        img_detect = cv2.copyTo(frame, None)
        if (self._map_detected):
            self._map_corner_axes = []
            # Create corner list and extract origin axes
            corners = np.concatenate([self._detected_markers[id][0] for id in self.MAP_MARKER_IDS])
            ret, rvec, tvec = cv2.solvePnP(self.MAP_MARKER_CORNERS, corners, self._cam_mtx, self._dist_coefs, cv2.SOLVEPNP_SQPNP)
            img_detect = cv2.drawFrameAxes(img_detect, self._cam_mtx, self._dist_coefs, rvec, tvec, 2, 4)
            self._map_corner_axes = (rvec, tvec)
        return img_detect
    
    def goalPose(self, frame: cv2.Mat) -> tuple[np.ndarray, cv2.Mat]:
        img_detect = cv2.copyTo(frame, None)
        if self._map_detected:
            corners = self._detected_markers[self.GOAL_MARKER_ID][0]
            # Compute transformation vectors & rotation vectors
            ret, rvec, tvec = cv2.solvePnP(self.GOAL_POSE_CORNERS, corners, self._cam_mtx, self._dist_coefs)
            img_detect = cv2.drawFrameAxes(img_detect, self._cam_mtx, self._dist_coefs, rvec, tvec, 2, 4)
            # Compute the location with referential of map
            if len(self._map_corner_axes) == 0:
                self.extractMapCornerPose(img_detect)
            self._goal_pose = self._convertToOriginReferential(self._map_corner_axes[0], self._map_corner_axes[1], tvec)[0,:2]
        return self._goal_pose, img_detect
    
    def thymioPose(self, frame: cv2.Mat, show_all=False) -> tuple[bool, np.ndarray, float, cv2.Mat]:
        img_detect = cv2.copyTo(frame, None)
        if not (self._map_detected and self._thymio_detected):
            return False, np.zeros(2), 0.0, img_detect
        # Draw heading vector
        v_thymio = np.array([[0,0,0], [0,10,0]], dtype=float)
        v_pixels, _ = cv2.projectPoints(v_thymio, np.mean([self._thymio_axes[0][0],self._thymio_axes[1][0]], 0), 
                                                np.mean([self._thymio_axes[0][1], self._thymio_axes[1][1]], 0), self._cam_mtx, self._dist_coefs)
        cv2.arrowedLine(img_detect, v_pixels[0][0].astype(int), v_pixels[1][0].astype(int), self.DETECTED_THYMIO_COLOR, 3)
        # Estimate position based on bottom left corner
        thymio_pose = np.zeros((2,3))
        for t in range(2):
            rvec, tvec = self._map_corner_axes
            rvec_thymio, tvec_thymio = self._thymio_axes[t]
            thymio_pose[t,:] = self._convertToOriginReferential(rvec, tvec, tvec_thymio)
            # Draw lines
            if show_all:
                v_map = np.array([[0,0,0]], dtype=float)
                v_pixels, _ = cv2.projectPoints(v_map, rvec_thymio, tvec_thymio, self._cam_mtx, self._dist_coefs)
                v_pixels2, _ = cv2.projectPoints(v_map, rvec, tvec, self._cam_mtx, self._dist_coefs)
                cv2.line(img_detect, v_pixels2[0][0].astype(int), v_pixels[0][0].astype(int), [0,0,255], 1, cv2.LINE_4)

        # Compute thymio angle based on ref (y_axis on corner 0)
        v_ref = thymio_pose[0] - thymio_pose[1]
        angle = np.arctan2(v_ref[1],-v_ref[0])
        print(thymio_pose)

        return True, np.mean(thymio_pose[:,:2], 0), angle, img_detect
    

    def _convertToOriginReferential(self, rvec_base, tvec_base, tvec_new):
        R = cv2.Rodrigues(rvec_base)[0]
        P = np.hstack((R, tvec_base))
        # Homogeneous matrix & point
        P = np.vstack((P, np.array([0,0,0,1])))
        p = np.vstack((tvec_new.reshape((3,1)), [1]))
        return np.linalg.solve(P,p)[0:3].T
    
    def createMap(self, frame: cv2.Mat, bl_marker=2, res=1) -> np.ndarray:
        ret, img_map = self.detectMap(frame)
        if self._map_detected:
            roi_points = []
            for i in range(4):
                roi_points.append(self._detected_markers[(i+bl_marker)%4][0][0])

            # Replace all detected markers with white square
            img_pre_project = cv2.copyTo(frame, None)
            color = cv2.mean(img_pre_project)[0:3]
            for corners in self._detected_markers.values():
                cv2.fillPoly(img_pre_project, corners.astype(int), color)
            # Find region of interest based on map corners
            p_mid_down = (roi_points[0] + roi_points[3])/2
            p_mid_up = (roi_points[1] + roi_points[2])/2
            roi_map = np.array([roi_points[0][0], roi_points[0][1], roi_points[3][0]-roi_points[0][0], p_mid_down[1] - p_mid_up[1]], dtype=int)
            # Project image
            x,y,w,h = roi_map
            m_perspective = cv2.getPerspectiveTransform(np.float32(roi_points), np.float32([[0,h],[0,0],[w,0],[w,h]]))
            img_project = cv2.warpPerspective(img_pre_project, m_perspective, (w,h))
            # Rotate image to have marker 0 in BL corner
            if bl_marker > 0:
                cv2.rotate(img_project, bl_marker-1, img_project)
            # Create figure to show progress
            fig = plt.figure(figsize=(20,8))
            fig.suptitle('Map extraction process')
            axes = fig.subplots(2,2)
            axes[0,0].axis('off')
            axes[0,0].set_title('1- Map delimitation')
            axes[0,0].imshow(img_map[:,:,::-1])
            axes[0,1].axis('off')
            axes[0,1].set_title('2- Retrieve projection')
            axes[0,1].imshow(img_project[:,:,::-1])

            # Apply Sobel filter to extract edges
            img = cv2.cvtColor(img_project, cv2.COLOR_BGR2GRAY)
            img_filtered = cv2.GaussianBlur(img, (11,11), 9)
            sobx = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, 3)
            soby = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, 3)
            sob = np.sqrt(sobx**2 + soby**2)
            sob = (sob * 255 / sob.max()).astype(np.uint8)
            # Thicken the edges
            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
            #sob = cv2.dilate(sob, kernel)
            # Apply threshold for binary transformation
            cv2.threshold(sob, self.THRESHOLD, 255, cv2.THRESH_BINARY, sob)
            #cv2.adaptiveThreshold(sob, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
            axes[1,0].axis('off')
            axes[1,0].set_title('3- Apply Sobel + Threshold')
            axes[1,0].imshow(sob, cmap='gray')

            # Create the grid based map
            # Find grid shape in pixel space
            self._gridmap = np.zeros(((int)(res*self._map_size[0]), (int)(res*self._map_size[1])))            
            h,w = sob.shape
            grid_w, grid_h = w/(res*self._map_size[0]), h/(res*self._map_size[1])
            print(f'Pixels per grid cell: {grid_w} x {grid_h}')
            for i in range(len(self._gridmap)):
                for j in range(len(self._gridmap[0])):
                    c_x = math.floor(i*grid_w) + grid_w/2
                    c_y = math.floor((len(self._gridmap[0])-1-j)*grid_h) + grid_h/2
                    im_rect = cv2.getRectSubPix(sob, ((int) (round(grid_w, 0)),(int) (round(grid_h,0))), (c_x,c_y))
                    self._gridmap[i,j] = -1 if cv2.sumElems(im_rect)[0] > cv2.mean(im_rect)[0] else 0
            self._configure_ax(axes[1,1], self._gridmap.shape[0], self._gridmap.shape[1], res)
            # Add the goal if detected
            cmap = colors.ListedColormap(['red', 'white'])
            if self._goal_detected:
                self._goal_pose, _ = self.goalPose(frame)
                goal_pose_grid = self._map_to_grid(self._goal_pose, res)
                print(self._goal_pose)
                goal_mask = (res, res)
                begin_x = max(goal_pose_grid[0] - goal_mask[0], 0)
                end_x = min(goal_pose_grid[0] + goal_mask[0], self._gridmap.shape[0])
                begin_y = max(goal_pose_grid[1] - goal_mask[1], 0)
                end_y = min(goal_pose_grid[1] + goal_mask[1], self._gridmap.shape[1])
                self._gridmap[begin_x:end_x, begin_y:end_y] = 1
                cmap = colors.ListedColormap(['red', 'white', 'green'])
            axes[1,1].imshow(self._gridmap.transpose(), cmap=cmap, origin='lower')
            axes[1,1].set_title('4- Extracted Grid Map')

        return self._gridmap
    
    def _map_to_grid(self, map_point, res) -> np.ndarray:
        return np.array(map_point*res, dtype=int)
    
    def _grid_to_map(self, grid_point, res) -> np.ndarray:
        return grid_point / res
    
    def _configure_ax(self, ax, max_x, max_y, res):
        """
        Helper function to create a figure of the desired dimensions & grid
        
        :param max_val: dimension of the map along the x and y dimensions
        :return: the fig and ax objects.
        """
        MAJOR = 10
        MINOR = 5
        
        major_ticks_x = np.arange(0, max_x+1, MAJOR)
        minor_ticks_x = np.arange(0, max_x+1, MINOR)
        major_ticks_y = np.arange(0, max_y+1, MAJOR)
        minor_ticks_y = np.arange(0, max_y+1, MINOR)
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.set_ylim([-1,max_y])
        ax.set_xlim([-1,max_x])
        ax.set_ylabel('y (cm)')
        ax.set_xlabel('x (cm)')
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/res))
        ax.xaxis.set_major_formatter(ticks_x)
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/res))
        ax.yaxis.set_major_formatter(ticks_y)
        ax.grid(True)

        return ax


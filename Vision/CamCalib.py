###########################################
#       CAMERA CALIBRATION CLASS
###########################################
#   INSPIRED FROM OPENCV TUTORIAL
#   --> https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html <--
###########################################

import cv2
import time
import glob
import numpy as np
import pandas as pd
from IPython.display import display

class CamCalib():

    def __init__(self, cam, chess_grid_size):
        self._cam = cam
        self._grid_size = chess_grid_size
        self.output_file_name = 'cal_'
        self.term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.index = 0

    
    def take_pictures(self, output_dir='Vision/images/cal'):
        # Loop until user presses 'q' key
        while True:
            # Take picture from cam
            if (self._cam is None):
                print("Error:: cam is None!")
                return
            elif (not self._cam.isOpened()):
                print("Error:: Cam not ready!")
                return

            ret, frame = self._cam.read()
            if ret:
                # Show detected chessboard if any
                ret, corners = cv2.findChessboardCorners(frame, self._grid_size, None)
                frame_copy = cv2.copyTo(frame,None)
                if ret:
                    frame_corners = cv2.cornerSubPix(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), self.term_criteria)
                    frame_copy = cv2.drawChessboardCorners(frame_copy, self._grid_size, frame_corners, ret)
                # Show image to user
                cv2.imshow('frame', frame_copy)
                key = cv2.waitKey(10)
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    # Save image to foler
                    dest = f'{output_dir}/{self.output_file_name}{self.index}.jpg'
                    ret = cv2.imwrite(dest, frame)
                    if not ret:
                        print("Could not write image to folder! --> ", dest)
                    else:
                        self.index += 1
            else:
                print("Error:: Could not read the cam frame!")
                time.sleep(1)
        
        cv2.destroyAllWindows()

    def calibrate(self, square_size, file_dir='Vision/images/cal'):
        i = 0
        file_list = glob.glob(f'{file_dir}/{self.output_file_name}*.jpg')
        objp = np.zeros((self._grid_size[0]*self._grid_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self._grid_size[0],0:self._grid_size[1]].T.reshape(-1,2)
        objp *= square_size
        obj_points_list = []
        image_points_list = []
        for f in file_list:
            im = cv2.imread(f)
            ret, corners = cv2.findChessboardCorners(im, self._grid_size, None)
            if ret:
                obj_points_list.append(objp)
                img_points = cv2.cornerSubPix(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), self.term_criteria)
                image_points_list.append(img_points)
                #Save marked image
                im_markers = cv2.drawChessboardCorners(im, (9,6), img_points, ret)
                cv2.imwrite(f'{file_dir}/marked_{i}.jpg', im_markers)
                i += 1
        # Perform calibration and obtain cam matrices
        h, w = im.shape[:2]
        ret, cam_mat, dist_mat, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, image_points_list, (w,h), None, None)
        print('Calibration success!')
        print('Camera matrix:\n[ ','\n'.join(['\t'.join([str(cell) for cell in row]) for row in cam_mat]), ' ]')
        df = pd.DataFrame(dist_mat, columns=['k1', 'k2', 'p1', 'p2', 'k3'])
        df = df.style.set_caption('Distortion params')
        display(df)
        # Save camera settings
        np.savez('data/calib_params', cam_mat, dist_mat, rvecs, tvecs)
        print("Calibration settings saved in ", 'files/calib_params')

    def undistord(self, frame: cv2.Mat, cam_mat, dist_coefs):
        w,h = frame.shape[0:2][::-1]
        img_borders = cv2.copyTo(frame, None)
        new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coefs, (w,h), 1, (w,h), 0)
        mapx, mapy = cv2.initUndistortRectifyMap(cam_mat, dist_coefs, None, new_cam_mat, (w,h), 5)
        img_borders = cv2.remap(img_borders, mapx, mapy, cv2.INTER_LINEAR)
        return img_borders, roi

    def load_camera_params(self, saved_dir='data'):
        """
        Return saved calibration parameters.

        Tuple -> cam_mtx, dist_coeffs, rvecs, tvecs
        """
        npz = np.load(f'{saved_dir}/calib_params.npz')
        cam_mat, dist_coefs, rvecs, tvecs = (npz[f'arr_{i}'] for i in range(4))
        npz.close()
        return cam_mat, dist_coefs, rvecs, tvecs


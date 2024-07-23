import cv2

class Camera:

    def  __init__(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise ValueError ("Camera Not Found")
        self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def __del__(self):
        self.camera.release()

    def getframe(self):
        if self.camera.isOpened():
            ret , frame = self.camera.read()

            if ret:
                return ret, frame
            else:
                return ret, None
        else:
            return None
    
    def release(self):
        if self.camera.isOpened():
            self.camera.release()

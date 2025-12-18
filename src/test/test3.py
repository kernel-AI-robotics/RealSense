import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.bridge = CvBridge()

        topic = '/camera/color/image_raw'   # 필요하면 바꿔
        self.sub = self.create_subscription(Image, topic, self.cb, 10)
        self.get_logger().info(f"Subscribing: {topic}")

    def cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, 50, 150)

        # 끊긴 선 연결
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 2000:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4 and area > best_area:
                best = approx.reshape(-1,2)
                best_area = area

        if best is not None:
            corners = order_corners(best)
            for (x,y) in corners.astype(int):
                cv2.circle(frame, (x,y), 6, (0,255,0), -1)
            cv2.polylines(frame, [corners.astype(int)], True, (0,255,0), 2)

            print("pixel corners (tl,tr,br,bl):", corners.tolist())

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            rclpy.shutdown()

def main():
    rclpy.init()
    node = ImageViewer()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

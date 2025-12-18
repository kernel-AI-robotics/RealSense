import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.bridge = CvBridge()

        topic = '/camera/color/image_raw'   # 필요하면 바꿔
        self.sub = self.create_subscription(Image, topic, self.cb, 10)
        self.get_logger().info(f"Subscribing: {topic}")

    def cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('RealSense Color', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
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

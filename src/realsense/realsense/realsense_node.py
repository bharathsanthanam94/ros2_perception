#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class MyNode(Node):
    def __init__(self):
        super().__init__("realsense")
        self.bridge = CvBridge()
        
        self.image_subscriber = self.create_subscription(Image, "/camera/realsense2_camera_node/color/image_raw", self.image_callback, 10)
        self.image_publisher = self.create_publisher(Image, "image_display", 10)
        self.counter=0
        self.get_logger().info("Realsense node has been started")
    def image_callback(self,msg):
        try:
           
           self.counter+=1
           if self.counter%30==0:
               self.get_logger().info(f'Received and publishing image #{self.counter}')
           self.image_publisher.publish(msg)
        
         
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
      


def main(args=None):
    rclpy.init(args=args)
    node =MyNode()
    # keep the node running
    rclpy.spin(node)
    #last line of the program
    rclpy.shutdown()
if __name__ == "__main__":
    main()
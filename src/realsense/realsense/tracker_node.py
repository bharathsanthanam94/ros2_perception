import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from realsense_msgs.msg import DetectionArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from realsense.tracker.byte_tracker import BYTETracker
from realsense.utils.visualize import plot_tracking

class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.30
        self.track_buffer = 30
        self.match_thresh = 0.9
        self.aspect_ratio_thresh = 10.0
        self.min_box_area = 1.0
        self.mot20 = False

class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")
        self.bridge = CvBridge()
        
        # Initialize tracker parameters
        tracker_args = TrackerArgs()
        
        # Initialize ByteTracker
        self.tracker = BYTETracker(tracker_args, frame_rate=30)
        
        # Create subscribers
        self.detection_sub = self.create_subscription(
            DetectionArray, 
            "detection_values", 
            self.detection_callback, 
            10
        )
        self.image_sub = self.create_subscription(
            Image,
            "image_display",
            self.image_callback,
            10
        )
        
        # Create publisher for tracked results
        self.tracking_pub = self.create_publisher(
            Image,
            "tracking_results",
            10
        )
        
        # Store the latest image
        self.current_image = None
        self.frame_id = 0
        
        self.get_logger().info("Tracker node has been started")

    def image_callback(self, msg: Image):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def detection_callback(self, msg: DetectionArray):
        if self.current_image is None:
            return

        try:
            # Reshape flattened detections back to original shape
            detections = np.array(msg.data).reshape(msg.shape)
            # log the detections shape 
            height, width = self.current_image.shape[:2]

            # Update tracker
            online_targets = self.tracker.update(
                torch.from_numpy(detections),
                [height, width],
                (640, 640)  # test_size
            )

            # Process tracking results
            online_tlwhs = []
            online_ids = []
            online_scores = []

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 10.0  # aspect_ratio_thresh
                
                if tlwh[2] * tlwh[3] > 1.0 and not vertical:  # min_box_area
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

            # Draw tracking results
            tracked_image = plot_tracking(
                self.current_image.copy(),
                online_tlwhs,
                online_ids,
                frame_id=self.frame_id,
                fps=30.0
            )

            # Publish tracking results
            tracking_msg = self.bridge.cv2_to_imgmsg(tracked_image, "bgr8")
            self.tracking_pub.publish(tracking_msg)
            
            self.frame_id += 1

        except Exception as e:
            self.get_logger().error(f"Error in tracking: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    
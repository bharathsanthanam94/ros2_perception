#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from realsense_msgs.msg import DetectionArray
from realsense_msgs.msg import Detection3DArray, Detection3D
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import numpy as np
import struct


class DepthNode(Node):
    def __init__(self):
        super().__init__("depth_node")
        self.bridge = CvBridge()
        
        # Initialize storage variables
        self.current_depth = None
        self.camera_intrinsics = None
        
        # Create subscribers
        self.detection_sub = self.create_subscription(
            DetectionArray,
            "detection_values",
            self.detection_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/realsense2_camera_node/aligned_depth_to_color/image_raw",
            self.depth_callback,
            5
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/realsense2_camera_node/color/camera_info",
            self.camera_info_callback,
            10
        )
        
        # Create publisher for 3D detections
        self.detection3d_pub = self.create_publisher(
            Detection3DArray,
            "detection_3d",
            10
        )
        
        # Replace marker publisher with point cloud publisher
        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            'detect_point',
            10
        )
        
        self.get_logger().info("Depth node has been started")

    def depth_callback(self, msg: Image):
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_intrinsics = {
            'fx': msg.k[0],  # Focal length x
            'fy': msg.k[4],  # Focal length y
            'cx': msg.k[2],  # Principal point x
            'cy': msg.k[5]   # Principal point y
        }

    def get_3d_coordinates(self, bbox_center, depth_value):
        """Convert pixel coordinates and depth to 3D coordinates."""
        if self.camera_intrinsics is None:
            return None

        # Using pinhole camera model for 3D projection
        x = (bbox_center[0] - self.camera_intrinsics['cx']) * depth_value / self.camera_intrinsics['fx']
        y = (bbox_center[1] - self.camera_intrinsics['cy']) * depth_value / self.camera_intrinsics['fy']
        z = depth_value

        # log the coordinates
        self.get_logger().info(f"3D coordinates: x={x}, y={y}, z={z}")
        return (x, y, z)

    def get_depth_at_point(self, x, y, w, h, window_ratio=0.4):
        """Get depth value for an object using a dynamic sampling window.
        
        Args:
            x, y: Top-left corner of bounding box
            w, h: Width and height of bounding box
            window_ratio: Ratio of bbox size to use for sampling (0-1)
        """
        if self.current_depth is None:
            return None

        # Calculate sampling window size based on bbox dimensions
        window_w = int(w * window_ratio)
        window_h = int(h * window_ratio)
        
        # Calculate center and sampling region
        center_x = int(x + w/2)
        center_y = int(y + h/2)
        
        # Define sampling region boundaries
        y_min = max(0, center_y - window_h//2)
        y_max = min(self.current_depth.shape[0], center_y + window_h//2)
        x_min = max(0, center_x - window_w//2)
        x_max = min(self.current_depth.shape[1], center_x + window_w//2)

        # Extract depth window
        depth_window = self.current_depth[y_min:y_max, x_min:x_max]
        
        # Filter out zeros and invalid values
        valid_depths = depth_window[np.where((depth_window > 0) & (depth_window < 10000))]
        
        if len(valid_depths) > 0:
            # Use a more robust statistical measure
            # Remove outliers using percentile method
            q1 = np.percentile(valid_depths, 25)
            q3 = np.percentile(valid_depths, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_depths = valid_depths[(valid_depths >= lower_bound) & (valid_depths <= upper_bound)]
            
            if len(filtered_depths) > 0:
                return float(np.median(filtered_depths))
        return None

    def create_point_cloud(self, detection3d_array: Detection3DArray) -> PointCloud2:
        # Create point cloud message
        self.get_logger().info(f"Creating point cloud for {len(detection3d_array.detections)} detections")
        point_cloud = PointCloud2()
        point_cloud.header = detection3d_array.header
        
        # Define fields for x, y, z coordinates and RGB color
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Define some distinct colors for different classes (RGB format)
        colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
        ]
        
        # Prepare point cloud data
        points = []
        for detection in detection3d_array.detections:
            # Convert coordinates from millimeters to meters
            x = detection.position.x / 1000.0
            y = detection.position.y / 1000.0
            z = detection.position.z / 1000.0
            
            # Get color based on class_id
            r, g, b = colors[detection.class_id % len(colors)]
            rgb = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
            
            points.append([x, y, z, rgb])
            
            # Log the converted coordinates
            self.get_logger().info(f"Point cloud coordinate (meters): x={x:.3f}, y={y:.3f}, z={z:.3f}")
        
        # Convert points to binary data
        point_data = []
        for point in points:
            point_data.extend([struct.pack('f', coord) for coord in point])
        
        # Set point cloud parameters
        point_cloud.height = 1
        point_cloud.width = len(points)
        point_cloud.fields = fields
        point_cloud.is_bigendian = False
        point_cloud.point_step = 16  # 4 * float32 (4 bytes each)
        point_cloud.row_step = point_cloud.point_step * point_cloud.width
        point_cloud.data = b''.join(point_data)
        point_cloud.is_dense = True
        
        return point_cloud

    def detection_callback(self, msg: DetectionArray):
        if self.current_depth is None or self.camera_intrinsics is None:
            self.get_logger().warn("Depth or camera info not available")
            return

        try:
            # Create Detection3DArray message
            detection3d_array = Detection3DArray()
            detection3d_array.header = msg.header  # Use the same header as input
            detection3d_array.header.frame_id = "camera_color_optical_frame"
            #log the detection array
            self.get_logger().info(f"Detection array: {msg}")
            # Get number of detections from shape
            num_detections = msg.shape[0]  # First dimension is number of detections
            values_per_detection = msg.shape[1]  # Second dimension is values per detection
            
            # Reshape the flattened data array into a 2D array
            detections = np.array(msg.data).reshape(num_detections, values_per_detection)
            
            # Process each detection
            for detection in detections:
                # Extract bounding box coordinates (x, y, w, h format)
                x, y, w, h = detection[:4]
                confidence = detection[4]
                class_id = int(detection[5])
                
                # Calculate center point of bounding box
                center_x = int(x + w/2)
                center_y = int(y + h/2)
                
                # Get depth at center point
                depth_value = self.get_depth_at_point(center_x, center_y, w, h)
                
                if depth_value is not None:
                    # Calculate 3D coordinates
                    coords_3d = self.get_3d_coordinates((center_x, center_y), depth_value)
                    
                    if coords_3d is not None:
                        detection3d = Detection3D()
                        detection3d.class_id = class_id
                        detection3d.score = float(confidence)
                        detection3d.position = Point(
                            x=float(coords_3d[0]),
                            y=float(coords_3d[1]),
                            z=float(coords_3d[2])
                        )
                        detection3d_array.detections.append(detection3d)

            # Publish 3D detection results
            self.detection3d_pub.publish(detection3d_array)
            self.get_logger().info(f"Published point cloud with{len(detection3d_array.detections)} points") 
            # Replace marker publishing with point cloud publishing
            point_cloud = self.create_point_cloud(detection3d_array)
            self.point_cloud_pub.publish(point_cloud)

        except Exception as e:
            self.get_logger().error(f"Error processing detections: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from realsense_msgs.msg import DetectionArray
from realsense.postprocess_frame import postprocess
from realsense.visualize import draw_detections, draw_fps
# COCO class names
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)
class DetectorNode(Node):
    def __init__(self,model_path):
        super().__init__("detector_node")
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #load model from the torch hub yolox_S, map location to GPU if available else CPU
        self.model = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_s", pretrained=True, device=self.device)
        #set model to evaluation mode
        self.model.eval()
        #log if the model is loaded on GPU or CPU
        self.get_logger().info(f"Model loaded on {self.device}")

        #set the confidence threshold and nms threshold
        self.confthre = 0.45
        self.nmsthre = 0.45
        self.num_classes = 80
    
        
        #create subscriber for image topic
        self.image_subscriber = self.create_subscription(Image, "image_display", self.image_callback, 10)
        self.detection_publisher= self.create_publisher(Image, "detection_results",10)
        self.detection_value_publisher = self.create_publisher(DetectionArray, "detection_values",10)
        self.preprocess_publisher= self.create_publisher(Image, "preprocess_results",10)
        self.counter =0
        self.get_logger().info("Detector node has been started")

    def preprocess_image(self, img, input_size):
        img, ratio = self.preproc(img, input_size)
        
        # Convert preprocessed image back to BGR for visualization
        vis_img = img.transpose(1, 2, 0).astype(np.uint8)
        # Publish the preprocessed image for visualization
        preprocess_msg = self.bridge.cv2_to_imgmsg(vis_img, "rgb8")
        self.preprocess_publisher.publish(preprocess_msg)
        
        # Convert to tensor
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        img = img.float()
        return img, ratio
    
    def preproc(self,img, input_size, swap=(2, 0, 1)):
    
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    
        return padded_img, r
    def image_callback(self, msg: Image):
        self.counter+=1
        if self.counter%30==0:
            self.get_logger().info(f"Received image #{self.counter}")
        try:
            #convert ros image to cv2 image in RGB format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            #convert to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            #preprocess the image
            img,ratio = self.preprocess_image(rgb_image, (640,640))

            #perform inference
            with torch.no_grad():
                outputs = self.model(img)
                self.get_logger().info(f"Outputs shape befor NMS: {outputs.shape}")
                #postprocess the outputs, perform NMS
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )
                            #log the shape of the outputs
                self.get_logger().info(f"Outputs shape after NMS: {outputs[0].shape}")
                result_image = rgb_image.copy()
                if outputs[0] is not None:
                    detections = outputs[0].detach()
                    result_image = draw_detections(result_image, detections, ratio,COCO_CLASSES,self.confthre)
                    #publish the detection values
                    detection_msg = DetectionArray()
                    detection_msg.header = msg.header
                    detection_msg.data = detections.cpu().numpy().flatten().tolist()
                    detection_msg.shape = [int(dim) for dim in detections.shape]
                    # print the shape of the detection_msg.datas
                    self.get_logger().info(f"Detection shape: {detection_msg.shape}")
                    self.detection_value_publisher.publish(detection_msg)


            #publish the original message for debugging
            detection_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            self.detection_publisher.publish(detection_msg)
        except Exception as e:
            self.get_logger().error(f"Error converting the image: {e}")
            return

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode(model_path="/home/bharath/ros2_ws/pretrained_models/yolox_s.pth")
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy


class DetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_detection_node")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', ''),
                ('conf_threshold', 0.3),
                ('image_topic', '/color/image_raw'),
                ('window_name', 'YOLO Detection')
            ]
        )
        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.image_topic = self.get_parameter('image_topic').value
        self.window_name = self.get_parameter('window_name').value

        if not self.model_path:
            self.get_logger().error("Model path parameter is required!")
            raise ValueError("Model path parameter is required!")

        # Initialize model
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"Successfully loaded YOLO model from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {str(e)}")
            raise

        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Image Subscriber
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile
        )
        self.get_logger().info(f"Subscribed to image topic: {self.image_topic}")

        # Display 
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self.get_logger().info(f"Created display window: {self.window_name}")

    def image_callback(self, msg):
        """Process incoming image messages."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                save=False,
                verbose=False
            )

            annotated_frame = self.annotate_image(frame, results)
            
            cv2.imshow(self.window_name, annotated_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.get_logger().info("Shutdown requested by user")
                self.cleanup()
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")


    def annotate_image(self, img, results):
        """Draw detection boxes and labels."""
        annotated_img = img.copy()
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                cv2.rectangle(
                    annotated_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),  # Green 
                    2  # Line thickness
                )

                label = f"Pallet: {conf:.2f}"
                label_size, _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    2
                )
                
                cv2.rectangle(
                    annotated_img,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1
                )
                
                cv2.putText(
                    annotated_img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),  # Black text
                    2
                )

        return annotated_img

    def cleanup(self):
        cv2.destroyAllWindows()
        self.get_logger().info("Cleaned up resources")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DetectionNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error running node: {str(e)}")
    finally:
        # Cleanup
        if 'node' in locals():
            node.cleanup()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np


class SegmentationNode(Node):
    def __init__(self):
        super().__init__("yolo_segmentation_node")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', ''),
                ('conf_threshold', 0.4),
                ('image_topic', '/color/image_raw'),
                ('window_name', 'YOLO Segmentation')
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

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f"Subscribed to image topic: {self.image_topic}")

        # Display
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self.get_logger().info(f"Created display window: {self.window_name}")

        self.colors = {
            0: (46, 139, 87),    # Green for floor
            1: (255, 69, 0),     # Red for pallet
            'overlay_alpha': 0.4,
            'contour_thickness': 2,
            'floor_contour': (0, 255, 0),
            'pallet_contour': (0, 69, 255)
        }

    def image_callback(self, msg):
        """Process incoming images."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                retina_masks=True,
                verbose=False
            )

            for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    visualization = self.create_visualization(frame, masks, classes, confidences)

                    cv2.imshow(self.window_name, visualization)

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.get_logger().info("Shutdown requested by user")
                self.cleanup()
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def create_visualization(self, image, masks, classes, confidences):
        """Create visualization with different colors per class."""
        overlay = image.copy()

        for mask, class_id, confidence in zip(masks, classes, confidences):
            mask = mask.astype(bool)

            colored_mask = np.zeros_like(image)
            colored_mask[mask] = self.colors[int(class_id)]

            overlay = cv2.addWeighted(
                overlay, 1,
                colored_mask,
                self.colors['overlay_alpha'],
                0
            )

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            contour_color = (self.colors['floor_contour'] 
                           if int(class_id) == 0 
                           else self.colors['pallet_contour'])

            cv2.drawContours(
                overlay,
                contours,
                -1,
                contour_color,
                self.colors['contour_thickness']
            )

            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                class_name = "Floor" if int(class_id) == 0 else "Pallet"
                text = f"{class_name} {confidence:.2f}"

                label_size, _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )
                cv2.rectangle(
                    overlay,
                    (cX - 60, cY - label_size[1] - 10),
                    (cX - 60 + label_size[0], cY),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    overlay,
                    text,
                    (cX - 60, cY - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        return overlay

    def cleanup(self):
        cv2.destroyAllWindows()
        self.get_logger().info("Cleaned up resources")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SegmentationNode()
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
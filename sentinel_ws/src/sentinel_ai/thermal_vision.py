import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class ThermalVisionNode(Node):
    def __init__(self, namespace):
        # STRIP THE SLASH BEFORE NAMING THE NODE
        safe_name = namespace.replace('/', '') if namespace else 'px4_0'
        super().__init__(f'thermal_vision_{safe_name}')
        
        # Keep the original namespace with the slash for subscribing to topics!
        self.namespace = namespace if namespace else ''
        self.bridge = CvBridge()
        
        self.get_logger().info(f"Loading YOLOv8 Model for {self.namespace}...")
        self.model = YOLO('/home/nicholas/sentinel_ws/src/sentinel_ai/best.pt')
        
        # --- THE FIX: DYNAMIC ID EXTRACTION ---
        # If namespace is '/px4_2', this removes '/px4_' and leaves '2'
        drone_id = self.namespace.replace('/px4_', '') if self.namespace else '0'
        
        # Construct the unique Gazebo camera topic for this specific drone
        cam_topic = f'/world/default/model/x500_depth_{drone_id}/link/camera_link/sensor/thermal_camera/image'
        self.subscription = self.create_subscription(Image, cam_topic, self.image_callback, 10)
            
        # Publish error to a NAMESPACED topic to prevent swarm cross-talk (Double slash fixed)
        self.error_pub = self.create_publisher(Point, f'{self.namespace}/target_error', 10)
        
        self.get_logger().info(f"SENTINEL AI {self.namespace} Active: Vision pipeline online.")

    def image_callback(self, msg):
        try:
            # 1. Grab raw data
            cv_image_16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            
            # 2. Standard 8-bit normalization
            cv_image_8 = cv2.normalize(cv_image_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # 3. FIX THE PHYSICS INVERSION: Gazebo renders the victims colder than the ground.
            cv_image_inverted = cv2.bitwise_not(cv_image_8)
            
            # 4. HIGH-POWER CONTRAST: Force the edges to pop
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
            cv_image_contrast = clahe.apply(cv_image_inverted)
            
            cv_image_rgb = cv2.cvtColor(cv_image_contrast, cv2.COLOR_GRAY2RGB)
            
            # 5. Prediction 
            results = self.model.predict(cv_image_rgb, conf=0.50, verbose=False)
            
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0].xyxy[0].cpu().numpy()
                confidence = float(results[0].boxes[0].conf[0].cpu().numpy())
                
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                img_h, img_w = cv_image_rgb.shape[:2]
                
                # Pixel Error
                err_x_pixels = cx - (img_w / 2.0)
                err_y_pixels = cy - (img_h / 2.0)
                
                # TRANSLATION
                meter_per_pixel = 0.05 
                err_y_meters = err_x_pixels * meter_per_pixel
                err_x_meters = -err_y_pixels * meter_per_pixel 
                
                error_msg = Point()
                error_msg.x = float(err_x_meters)
                error_msg.y = float(err_y_meters)
                error_msg.z = confidence 
                self.error_pub.publish(error_msg)
            
            annotated_frame = results[0].plot()
            img_h, img_w = cv_image_rgb.shape[:2]
            cv2.drawMarker(annotated_frame, (int(img_w/2), int(img_h/2)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            cv2.imshow(f"SENTINEL AI - Thermal Feed ({self.get_name()})", annotated_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Vision Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    import sys
    
    namespace = ''
    for arg in sys.argv[1:]:
        if arg.startswith('/px4_'):
            namespace = arg
            break
            
    node = None
    try:
        clean_name = namespace.replace('/', '') if namespace else 'px4_0'
        node = ThermalVisionNode(namespace)
        rclpy.spin(node)
    except Exception as e:
        print(f"Vision Error: {e}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

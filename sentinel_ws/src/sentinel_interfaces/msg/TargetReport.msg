# One thermal detection geolocated to the Gazebo/world ENU frame.
std_msgs/Header header
uint8 uav_id
float64 x                 # world ENU [m]
float64 y                 # world ENU [m]
float32 confidence        # detector confidence in [0, 1]
float32 slant_range       # camera-to-target distance [m]
float64 uav_x             # observer position (world ENU) at capture time
float64 uav_y
float64 uav_z

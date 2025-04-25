#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time
import os

from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

import numpy as np

# Import your planning functions
from robotic_arm_planner.planner_lib.closed_form_algorithm import closed_form_algorithm
# from robotic_arm_planner.planner_lib.path_planning import world_to_grid, grid_to_world
from robotic_arm_planner.planner_lib.Astar3D import find_path, world_to_grid, grid_to_world, dilate_obstacles
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import binary_dilation

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        
        # --- Parameters for the grid / reachability ---
        self.robot_name = 'ur10e'
        self.filename = "reachability_map_27_fused"
        fn_npy = f"{self.filename}.npy"
        self.grid_size = int(self.filename.split('_')[2])
        base_home = os.path.expanduser('~')
        self.reachability_map_fn = os.path.join(
            base_home,
            'ws_reachability',
            'rm4d',
            'experiment_scripts',
            'data',
            f'eval_poses_{self.robot_name}',
            fn_npy
        )
        self.get_logger().info(f"Loading reachability map from: {self.reachability_map_fn}")
        self.reachability_map = np.load(self.reachability_map_fn, allow_pickle=True).item()
        self.radius = 1.35  # For UR10e (hardcoded!!)

        # Create grid space
        self.resolution = 2*self.radius / self.grid_size
        self.x_vals = np.linspace(-self.radius + (self.resolution / 2), self.radius - (self.resolution / 2), self.grid_size)
        self.y_vals = self.x_vals
        self.z_levels = sorted(self.reachability_map.keys())
        self.grid_shape = (self.grid_size, self.grid_size, len(self.z_levels))
        z_min = min(self.z_levels)
        z_max = max(self.z_levels)
        self.z_vals = np.linspace(z_min, z_max, len(self.z_levels))  # assumes uniform spacing

        # Create subscriber and publishers
        self.create_subscription(Pose, '/goal_pose', self.goal_callback, 10)
        self.joint_pub = self.create_publisher(JointState, '/planned_joint_states', 10)
        self.status_pub = self.create_publisher(Bool, '/planner_success', 10)

        self.get_logger().info("Planner node initialized and waiting for goal poses...")

    def goal_callback(self, msg: Pose):
        goal_pos = [msg.position.x, msg.position.y, msg.position.z]
        self.get_logger().info(f"Received goal pose: {goal_pos}")

        # Example start position (NEEDS TO be parameterized)
        start_pos = [-0.25, 0.6, 0.2]
        start_orientation = R.from_euler('xyz', [60, 120, 150], degrees=True)
        goal_orientation = R.from_euler('xyz', [0, -90, 0], degrees=True)

        try:
            start_idx = world_to_grid(*start_pos, self.x_vals, self.y_vals, self.z_vals)
            goal_idx = world_to_grid(*goal_pos, self.x_vals, self.y_vals, self.z_vals)
        except Exception as e:
            self.get_logger().error(f"Failed to convert world to grid: {e}")
            self.status_pub.publish(Bool(data=False))
            return

        # Occupancy grid: all free (NEEDS TO be parameterized)
        occupancy_grid = np.zeros(self.grid_shape, dtype=np.uint8)
        # Obstacles
        # 3D meshgrid of coordinates
        X, Y, Z = np.meshgrid(self.x_vals, self.y_vals, self.z_vals, indexing='ij')  # shape: (grid_size, grid_size, num_z)

        # Define cube bounds (in world coordinates)
        cube_center = (0.2, 0.2, 1)  # (x, y, z)
        cube_size = 0.4  # length of each side

        # Compute bounds
        x0, y0, z0 = cube_center
        half = cube_size / 2

        cube_mask = (
            (X >= x0 - half) & (X <= x0 + half) &
            (Y >= y0 - half) & (Y <= y0 + half) &
            (Z >= z0 - half) & (Z <= z0 + half)
        )
        occupancy_grid[cube_mask] = 1

        # Define sphere parameters
        sphere_center = [(-0.3, 0.0, 0.4), (0.3, 0.5, 0.4)]
        sphere_radius = [0.3, 0.3]

        # Create mask for sphere
        for i in range(len(sphere_radius)):
            dist_squared = (X - sphere_center[i][0])**2 + (Y - sphere_center[i][1])**2 + (Z - sphere_center[i][2])**2
            sphere_mask = dist_squared <= sphere_radius[i]**2
            occupancy_grid[sphere_mask] = 1

        # Define the dilation distance (in meters, for example)
        dilation_distance = 0.4  # Enlarge obstacles by 0.1 meters in all directions

        # Apply dilation
        occupancy_grid_dilated = dilate_obstacles(occupancy_grid, dilation_distance, self.x_vals)

        path = find_path(occupancy_grid_dilated, start_idx, goal_idx)
        if not path:
            self.get_logger().warn("No path found!")
            self.status_pub.publish(Bool(data=False))
            return

        # Convert path to world coordinates
        path_world = [grid_to_world(i, j, k, self.x_vals, self.y_vals, self.z_vals) for i, j, k in path]

        # Interpolate orientations along the path
        key_rots = R.from_quat([start_orientation.as_quat(), goal_orientation.as_quat()])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        times = np.linspace(0, 1, len(path))
        interp_rots = slerp(times)
        interp_rot_matrices = interp_rots.as_matrix()

        # Convert to Euler angles
        interpolated_euler_angles = interp_rots.as_euler('xyz', degrees=True)
        print("Interpolated Euler angles:\n", interpolated_euler_angles)

        # Plan joint values
        home_position = np.array([0.0, -1.2, -2.3, -1.2, 1.57, 0.0])
        all_joint_values = []
        q_current = closed_form_algorithm(create_pose_matrix(path_world[0], interp_rot_matrices[0]), home_position, type=0)
        all_joint_values.append(q_current)

        for i in range(1, len(path_world)):
            T = create_pose_matrix(path_world[i], interp_rot_matrices[i])
            q_new = closed_form_algorithm(T, q_current, type=0)
            all_joint_values.append(q_new)
            q_current = q_new
            if np.any(np.isnan(q_new)):

                self.get_logger().error(f"Invalid IK at step {i}: pose = {T[:3, 3]}. Path wolrd = {path_world[i]}")

        # Publish joint values step-by-step
        for i, q in enumerate(all_joint_values):
            msg = JointState()
            msg.name = [f'joint_{j+1}' for j in range(len(q))]
            msg.position = q.tolist()
            msg.header.stamp = self.get_clock().now().to_msg()
            self.joint_pub.publish(msg)
            self.get_logger().info(f"Published step {i}: {np.round(q, 3)}")
            time.sleep(0.1)  # 10 Hz

        # Publish success
        self.status_pub.publish(Bool(data=True))
        self.get_logger().info("Joint planning complete and published.")

def create_pose_matrix(position, rotation_matrix):
    """Helper to create 4x4 transformation matrix from position and rotation matrix."""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position
    return T

def main(args=None):
    rclpy.init(args=args)
    planner_node = PlannerNode()
    try:
        rclpy.spin(planner_node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        print("Keyboard interrupt received. Shutting down planner node.")
    except Exception as e:
        print(f"Unhandled exception: {e}")

if __name__ == "__main__":
    main()
# MPPI with Human Motion Prediction

This is the control code used in the user study for the paper [How Human Motion Prediction Quality Shapes Social Robot Navigation Performance in Constrained Spaces](https://arxiv.org/abs/2601.09856), appearing at the 2026 International Conference on Human Robot Interaction (HRI).

## Getting Started

Be in a ROS2 environment (e.g., the `fluentrobotics/ros:humble` Docker image).

Run `poetry install`.

## Using the Simulator Node

In a dedicated terminal, run `rviz2 -d debug.rviz`

Make sure the virtual environment is activated: `poetry shell`

Run `python3 -m stretch_mppi.stretch_simulator`

Use the `2D Pose Estimate` tool at the top of the RViz window to override Stretch's pose (click and drag in the map to set a position and orientation).

## Running the MPPI Node

Make sure the virtual environment is activated: `poetry shell`

Run `python3 -m stretch_mppi.prototype_node`

## Running the HST or CoHAN Prediction Models

To change the prediction model, change the `METHOD` parameter in `controller_config.py`. You will need to set up and concurrently run the [HST Combo](https://github.com/fluentrobotics/hst_combo) or [CoHAN](https://github.com/LAAS-HRI/CoHAN2.0) in parallel.

## Using the Stretch

The rviz command for the simulator is actually still useful to visualize rollouts, but the state history won't be visualized without running the simulator.

Boot and home the robot.

Launch the stretch driver in navigation mode: `ros2 launch stretch_core stretch_driver.launch.py mode:=navigation`

Set up motion capture on the dedicated desktop, reconfigure settings on the Stretch if necessary, then run `ros2 launch mocap_optitrack_client launch_z_up.py`

# Model Team Documentation
## Pipeline Overview
The model team is responsible for the tracking of people (PEV riders) infront of the drone.
- The pipeline starts by feeding the drone's camera information to the `kestrel_vision` ROS2 package.
- `kestrel_vision` contains `vision_node.py`, which creates the ROS2 node that subscribes to the camera, runs YOLO and our CNN, and publishes these detections (bounding box + feature embeddings).
- Next, `kestrel_tracking` contains `tracking_node.py`, which similarly creates the node that subscribes to the vision node's detections, and begins tracking them. This involves initiating a TrackManager and publishing the associated Tracks.

## Pipeline Specifics
### kestrel_vision

## ROS Topics/Services

## Key Parameters

## Debug Checklist

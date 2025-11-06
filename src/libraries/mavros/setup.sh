#!/bin/bash

echo !!!!! Installing geoids to the correct path !!!!!
sudo bash ./mavros/scripts/install_geographiclib_datasets.sh
echo !!!!! Installing Mavros Proper !!!!!
colcon build --symlink-install --executor
echo !!!!! Testing Mavros Proper !!!!!
colcon test

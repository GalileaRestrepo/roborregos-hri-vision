#!/bin/bash
eval "$(/Users/agrm/miniconda3/bin/conda shell.bash hook)"
conda activate ros2_hri
export PYTHONPATH="/Users/agrm/ros2_hri_ws/install/hri_audio/lib/python3.9/site-packages:/Users/agrm/ros2_hri_ws/install/hri_interfaces/lib/python3.9/site-packages:/Users/agrm/ros2_hri_ws/install/hri_rag/lib/python3.9/site-packages:/Users/agrm/ros2_hri_ws/install/hri_vision/lib/python3.9/site-packages:${PYTHONPATH}"
export AMENT_PREFIX_PATH="/Users/agrm/ros2_hri_ws/install:${AMENT_PREFIX_PATH}"
python -m hri_rag.document_ingestion "$@"

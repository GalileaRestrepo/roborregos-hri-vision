#!/bin/bash
# compila hri_interfaces con los compiladores correctos de macOS
eval "$(/Users/agrm/miniconda3/bin/conda shell.bash hook)"
conda activate ros2_hri

PYTHON=$(/Users/agrm/miniconda3/envs/ros2_hri/bin/python -c "import sys; print(sys.executable)")

colcon build \
  --base-paths src \
  --build-base build \
  --install-base install \
  --packages-select hri_interfaces \
  --cmake-args \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
    -DPython3_EXECUTABLE=$PYTHON \
    -DPython3_INCLUDE_DIR=$(/Users/agrm/miniconda3/envs/ros2_hri/bin/python -c "import sysconfig; print(sysconfig.get_path('include'))") \
    -DPython3_LIBRARY=$(/Users/agrm/miniconda3/envs/ros2_hri/bin/python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

echo "hri_interfaces compilado"

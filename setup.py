"""Setup script for the pinger_localization ROS 2 package (ament_python)."""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = "pinger_localization"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # ament_index marker so ros2 can locate the package/share directory.
        (
            os.path.join("share", "ament_index", "resource_index", "packages"),
            [os.path.join("resource", package_name)],
        ),
        # Manifest + share data (launch files, config) under share/<pkg>.
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tuanpham",
    maintainer_email="tuanpham@todo.todo",
    description=(
        "Underwater acoustic pinger 2D localization: simulator, six solver "
        "algorithms, and evaluation tools."
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            # Names keep the ".py" suffix so existing launch files and
            # `ros2 run pinger_localization <node>` references work unchanged.
            "pinger_simulator.py = pinger_localization.pinger_simulator:main",
            "eval_node.py = pinger_localization.eval_node:main",
            "batch_sliding.py = pinger_localization.solvers.batch_sliding:main",
            "batch_full.py = pinger_localization.solvers.batch_full:main",
            "particle_filter.py = pinger_localization.solvers.particle_filter:main",
            "iterative_ekf.py = pinger_localization.solvers.iterative_ekf:main",
            "iterative_rls.py = pinger_localization.solvers.iterative_rls:main",
            "iterative_gd.py = pinger_localization.solvers.iterative_gd:main",
            "sonar_trace_node.py = pinger_localization.sonar_trace_node:main",
        ],
    },
)

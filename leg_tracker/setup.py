import os
from glob import glob
from setuptools import setup

package_name = "leg_tracker"

setup(
    # Other parameters ...
    data_files=[
        # ... Other data files
        # Include all launch files.
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        )
    ]
)

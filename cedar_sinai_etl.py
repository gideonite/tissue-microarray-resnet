"""
Handles the ETL for Ciders-Sinai dataset.
"""

import os
path = os.path.expandvars("$TMPDIR/TIFF color normalized sequential filenames/test100.tif")

with open(path) as f:
    print(f)

import time

import fastmunk
import munkres
import numpy as np

from leaguescheduler.constants import DISALLOWED_NBR

mat_clean = np.array(
    [
        [12, 9, 27, 10, 23],
        [7, 13, 13, 30, 19],
        [25, 18, 26, 11, 26],
        [9, 28, 26, 23, 13],
        [16, 16, 24, 6, 9],
    ],
    dtype=np.float64,
)

################
## CLEAN #######
################

m = fastmunk.FastMunk()
start = time.time()
for _ in range(10000):
    indices_clean_fm = m.compute(mat_clean)
print("FastMunk (clean): ", time.time() - start, "[s]", "indices =>", indices_clean_fm)

m = munkres.Munkres()
start = time.time()
for _ in range(10000):
    indices_clean_mk = m.compute(mat_clean)
print("Munkres (clean): ", time.time() - start, "[s]", "indices =>", indices_clean_mk)

print("Is equal?", indices_clean_fm == indices_clean_mk)

###############
## DIRTY ######
###############

mat_dirty = np.array(
    [
        [12, DISALLOWED_NBR, 27, 10, 23],
        [7, DISALLOWED_NBR, 13, 30, 19],
        [25, 18, 26, 11, 26],
        [9, 28, DISALLOWED_NBR, 23, 13],
        [16, 16, 24, 6, 9],
    ],
    dtype=np.float64,
)

m = fastmunk.FastMunk()
start = time.time()
for _ in range(10000):
    indices_dirty_fm = m.compute(mat_dirty)
print("FastMunk (dirty): ", time.time() - start, "[s]", "indices =>", indices_dirty_fm)

mat_dirty = [
    [12, munkres.DISALLOWED, 27, 10, 23],
    [7, munkres.DISALLOWED, 13, 30, 19],
    [25, 18, 26, 11, 26],
    [9, 28, munkres.DISALLOWED, 23, 13],
    [16, 16, 24, 6, 9],
]

m = munkres.Munkres()
start = time.time()
for _ in range(10000):
    indices_dirty_mk = m.compute(mat_dirty)
print("Munkres (dirty): ", time.time() - start, "[s]", "indices =>", indices_dirty_mk)

print("Is equal?", indices_dirty_fm == indices_dirty_mk)

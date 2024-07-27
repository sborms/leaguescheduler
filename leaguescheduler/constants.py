LARGE_NBR = 999
DISALLOWED_NBR = -99

# 0 --> e.g., a game on Monday and Tuesday (0 rest days but delta t is 1)
DICT_REST_DAYS = {
    0: 1000,
    1: 400,
    2: 160,
    3: 64,
    4: 26,
    18: 26,
    5: 10,
    17: 10,
    6: 4,
    16: 4,
    7: 2,
    15: 2,
    8: 1,
    14: 1,
}

# NOTE: The order of columns should stay date - time - location - home - away!
OUTPUT_COLS = ["Date", "Hour", "Location", "Home", "Away"]

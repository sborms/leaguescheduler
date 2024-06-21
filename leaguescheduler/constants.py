LARGE_NBR = 999
DISALLOWED_NBR = -99

# cf. slightly modified from original penalisation table
# 0 --> e.g., a game on Monday and Tuesday (0 rest days but delta t is 1)
DICT_REST_DAYS = {
    0: 750,
    1: 500,
    2: 250,
    3: 100,
    4: 50,
    18: 50,
    5: 30,
    17: 30,
    6: 20,
    16: 20,
    7: 15,
    15: 15,
    8: 10,
    14: 10,
    # 9: 5,
    # 13: 5,
}

# NOTE: The order of columns should stay date - time - location - home - away!
OUTPUT_COLS = ["Date", "Hour", "Location", "Home", "Away"]

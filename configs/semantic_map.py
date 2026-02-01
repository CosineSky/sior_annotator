SEMANTIC_MAP = {
    "background": 0,
    "road": 1,
    "railway": 2,
    "bridge": 3,
    "airport": 4,
    "runway": 5,
    "harbor": 6,
    "ship": 7,
    "airplane": 8,
    "vehicle": 9,
    "building": 10,
    "industrial_area": 11,
    "residential_area": 12,
    "farmland": 13,
    "forest": 14,
    "grassland": 15,
    "bare_land": 16,
    "river": 17,
    "lake": 18,
    "sea": 19
}

ID2SEMANTIC = {v: k for k, v in SEMANTIC_MAP.items()}
NUM_CLASSES = len(SEMANTIC_MAP)

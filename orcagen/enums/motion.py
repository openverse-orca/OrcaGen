from enum import Enum


class MotionType(str, Enum):
    STATIC = "static"
    FREE_FALL = "free_fall"
    ROLLING = "rolling"
    SLIDING = "sliding"
    ROTATING_IN_PLACE = "rotating_in_place"
    OSCILLATING = "oscillating"
    PENDULUM = "pendulum"
    UNIFORM_LINEAR = "uniform_linear"
    UNKNOWN = "unknown"


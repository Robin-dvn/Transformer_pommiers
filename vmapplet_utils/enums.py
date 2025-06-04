from enum import Enum, IntEnum


class Observation(str, Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    SYLLEPTIC_SMALL = "SYLLEPTIC_SMALL"
    SYLLEPTIC_MEDIUM = "SYLLEPTIC_MEDIUM"
    SYLLEPTIC_LARGE = "SYLLEPTIC_LARGE"
    DORMANT = "DORMANT"
    FLORAL = "FLORAL"
    TRUNK = "TRUNK"
    NEW_SHOOT = "NEW_SHOOT"


class Zone(IntEnum):
    """
    Region of the sequence of nodes in a branch with similar lateral branching properties
    Note: Markov has a 7th "absorbing/final" state
    """

    DORMANT_START = 0
    SMALL = 1
    DIFFUSE = 2
    MEDIUM = 3
    FLORAL = 4
    DORMANT_END = 5


class FruitState(str, Enum):
    FLOWER = "FLOWER"
    NO_FLOWER = "NO_FLOWER"
    FRUIT_SCAR = "FRUIT_SCAR"
    FRUIT = "FRUIT"


class LeafState(str, Enum):
    SCAR = "SCAR"
    GROWING = "GROWING"

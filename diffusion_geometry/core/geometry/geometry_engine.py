from abc import ABC


class GeometryEngine(ABC):
    """
    Abstract base class for geometry engines that compute geometric objects.
    """

    def __init__(self, xp=None):
        import numpy

        self.xp = xp if xp is not None else numpy

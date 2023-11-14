"""flyact: leak-free video classification of Drosophila micro-behaviours."""

CLASSES = ("feeding", "grooming", "pumping")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

__version__ = "0.1.0"

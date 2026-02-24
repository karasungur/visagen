"""
Face Type definitions for Visagen.

Defines different face crop types with their padding values.
"""

from enum import IntEnum


class FaceType(IntEnum):
    """
    Face type enumeration defining crop regions.

    Each type has an associated padding value that determines
    how much area around the face is included in the crop.

    Attributes:
        HALF: Minimal face crop (padding: 0.0)
        MID_FULL: Medium face crop (padding: 0.0675)
        FULL: Full face with some context (padding: 0.2109375)
        FULL_NO_ALIGN: Full face without rotation alignment (padding: 0.2109375)
        WHOLE_FACE: Entire face with forehead (padding: 0.40)
        HEAD: Full head including hair (padding: 0.70)
        HEAD_NO_ALIGN: Full head without rotation alignment (padding: 0.70)
    """

    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    HEAD = 5
    HEAD_NO_ALIGN = 6

    @property
    def padding(self) -> float:
        """Get the padding value for this face type."""
        padding_map = {
            FaceType.HALF: 0.0,
            FaceType.MID_FULL: 0.0675,
            FaceType.FULL: 0.2109375,
            FaceType.FULL_NO_ALIGN: 0.2109375,
            FaceType.WHOLE_FACE: 0.40,
            FaceType.HEAD: 0.70,
            FaceType.HEAD_NO_ALIGN: 0.70,
        }
        return padding_map.get(self, 0.0)

    @property
    def remove_align(self) -> bool:
        """Check if this face type should skip rotation alignment."""
        return self in (FaceType.FULL_NO_ALIGN, FaceType.HEAD_NO_ALIGN)

    @classmethod
    def from_string(cls, name: str) -> "FaceType":
        """
        Create FaceType from string name.

        Args:
            name: Face type name (case-insensitive).

        Returns:
            Corresponding FaceType enum value.

        Raises:
            ValueError: If name is not a valid face type.
        """
        name_upper = name.upper().replace(" ", "_")
        try:
            return cls[name_upper]
        except KeyError:
            valid = ", ".join(ft.name for ft in cls)
            raise ValueError(f"Unknown face type '{name}'. Valid types: {valid}")

    def to_string(self) -> str:
        """Convert to lowercase string representation."""
        return self.name.lower()

    def get_padding_and_align(self) -> tuple[float, bool]:
        """
        Get padding value and alignment flag.

        Returns:
            Tuple of (padding, remove_align).
        """
        return self.padding, self.remove_align


# Mapping for compatibility
FACE_TYPE_TO_PADDING = {
    FaceType.HALF: (0.0, False),
    FaceType.MID_FULL: (0.0675, False),
    FaceType.FULL: (0.2109375, False),
    FaceType.FULL_NO_ALIGN: (0.2109375, True),
    FaceType.WHOLE_FACE: (0.40, False),
    FaceType.HEAD: (0.70, False),
    FaceType.HEAD_NO_ALIGN: (0.70, True),
}

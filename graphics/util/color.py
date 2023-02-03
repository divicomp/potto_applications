from dataclasses import dataclass


@dataclass(frozen=True)
class Color:
    r: float
    g: float
    b: float
    a: float

    def __add__(self, other: 'Color') -> 'Color':
        return Color(self.r + other.r,
                     self.g + other.g,
                     self.b + other.b,
                     self.a + other.a)

    def __mul__(self, other: 'Color') -> 'Color':
        return Color(self.r * other.r,
                     self.g * other.g,
                     self.b * other.b,
                     self.a * other.a)

    def __truediv__(self, other: float) -> 'Color':
        return Color(self.r / other,
                     self.g / other,
                     self.b / other,
                     self.a / other)

    def tolist(self) -> list[float]:
        return [self.r, self.g, self.b, self.a]


@dataclass
class Colors:
    BLACK = Color(0, 0, 0, 1)
    WHITE = Color(1, 1, 1, 1)
    RED = Color(1, 0, 0, 1)
    GREEN = Color(0, 1, 0, 1)
    BLUE = Color(0, 0, 1, 1)
    TRANSPARENT = Color(0, 0, 0, 0)

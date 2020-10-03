class Color:
    def __init__(self, r, g, b, t=10):
        self.rgb = (r, g, b)
        self.t = t

    def __eq__(self, otherrgb):
        if isinstance(otherrgb, Color):
            otherrgb = otherrgb.rgb
        for c1, c2 in zip(self.rgb, otherrgb):
            if abs(c1 - c2) > self.t:
                return False
        return True

class COLORS:
    ORANGE = Color(240, 176, 96)
    RED = Color(238, 34, 17)
    BLACK_STONE = Color(44, 44, 44)
    WHITE_STONE = Color(243, 243, 243)
    PLAYING_TRIANGLE = Color(31, 41, 47)
    WHITE = Color(255, 255, 255)
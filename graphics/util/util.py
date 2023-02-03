

def lerp(t, a, b):
    return (1 - t) * a + t * b


def unlerp(s, a, b):
    return (s - a) / (b - a)


def remap(v, lohi_old, lohi_new):
    lo0, hi0 = lohi_old
    lo1, hi1 = lohi_new
    return lerp(unlerp(v, lo0, hi0), lo1, hi1)


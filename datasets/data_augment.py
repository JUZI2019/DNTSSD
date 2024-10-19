from numpy import random


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img


class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast()
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()

    def __call__(self, img):
        img = self.rb(img)
        if random.randint(2):
            distort = self.pd
        else:
            distort = self.pd
        img = distort(img)
        img = self.rln(img)
        return img


class RandomBrightness2(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, bg):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
            bg += delta
        return img, bg
    
class RandomContrast2(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, bg):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
            bg *= alpha
        return img, bg

class SwapChannels2(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img, bg):
        img = img[:, :, self.swaps]
        bg = bg[:, :, self.swaps]

        return img, bg

class RandomLightingNoise2(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img, bg):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels2(swap)
            img, bg = shuffle(img, bg)
        return img, bg

class PhotometricDistort2(object):
    def __init__(self):
        self.pd = RandomContrast2()
        self.rb = RandomBrightness2()
        self.rln = RandomLightingNoise2()

    def __call__(self, img, bg):
        img, bg = self.rb(img, bg)

        if random.randint(2):
            distort = self.pd
        else:
            distort = self.pd

        img, bg = distort(img, bg)

        img, bg = self.rln(img, bg)
        return img, bg

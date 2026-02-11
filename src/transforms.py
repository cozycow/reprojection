import numpy as np
from abc import ABC, abstractmethod


class Transform(ABC):

    @abstractmethod
    def __call__(self, r, alpha=1):
        pass

    @abstractmethod
    def __invert__(self):
        pass

    @property
    def isnull(self):
        return isinstance(self, Pipe) and len(self) == 0

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.__dict__ == other.__dict__

    def __add__(self, other):
        if self == ~other:
            return Pipe()
        elif not isinstance(other, Pipe):
            return Pipe(self, other)
        else:
            ### let the Pipe class decide
            return NotImplemented

    def __sub__(self, other):
        return self + ~other

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other


class Pipe(Transform, list):

    def __init__(self, *items):
        super().__init__(items)

    def __call__(self, r, alpha=1):
        r_, alpha_ = r, alpha
        for transform in self:
            r_, alpha_ = transform(r_, alpha_)
        return r_, alpha_

    def __getitem__(self, ii):
        item = super().__getitem__(ii)
        if isinstance(item, list):
            return type(self)(*item)
        else:
            return item

    def __invert__(self):
        return type(self)(*(~transform for transform in self[::-1]))

    def __add__(self, other):
        if other.isnull:
            return self
        elif self.isnull:
            return other
        elif isinstance(other, type(self)):
            if isinstance(other[0], type(self[-1])):
                return self[:-1] + (self[-1] + other[0]) + other[1:]
            else:
                return type(self)(*self, *other)
        else:
            return self[:-1] + (self[-1] + other)

    def __radd__(self, other):
        if self.isnull:
            return other
        else:
            return (other + self[0]) + self[1:]


class Translate(Transform):

    def __new__(cls, *args, **kwargs):
        if all(x == 0 for x in args[0]):
            return Pipe()
        else:
            return super().__new__(cls)

    def __init__(self, shift):
        self.shift = shift

    def __repr__(self):
        return f'Translate(shift:{self.shift})'

    def __call__(self, r, alpha=1):
        return tuple(x + dx for x, dx in zip(r, self.shift)), alpha

    def __invert__(self):
        return type(self)(tuple(-x for x in self.shift))

    def __add__(self, other):
        if isinstance(other, type(self)):
            return type(self)(tuple(x + y for x, y in zip(self.shift, other.shift)))
        else:
            return super().__add__(other)


class Scale(Transform):

    def __new__(cls, *args, **kwargs):
        if args[0] == 1:
            return Pipe()
        else:
            return super().__new__(cls)

    def __init__(self, factor):
        self.factor = factor

    def __repr__(self):
        return f'Scale(factor:{self.factor})'

    def __call__(self, r, alpha=1):
        return tuple(self.factor * x for x in r), alpha

    def __invert__(self):
        return type(self)(1 / self.factor)

    def __add__(self, other):
        if isinstance(other, type(self)):
            return type(self)(self.factor * other.factor)
        else:
            return super().__add__(other)


class Rotate(Transform):

    def __new__(cls, *args, **kwargs):

        if not isinstance(args[0], np.ndarray) and args[0] == 1:
            return Pipe()
        else:
            return super().__new__(cls)


    def __init__(self, *args):
        self.w = args[0]
        self.u = args[1:]

    @property
    def axis(self):
        q = np.sqrt(sum(x ** 2 for x in self.u))
        return tuple(float(x / q) for x in self.u)

    @property
    def angle(self):
        q = np.sqrt(sum(x ** 2 for x in self.u))
        return float(2 * np.arctan2(q, self.w) * np.pi / 180)

    def __repr__(self):
        return f'Rotate(axis:{self.axis}, angle:{self.angle})'

    def __mul__(self, other):
        return type(self)(self.w * other.w - sum(x * y for x, y in zip(self.u, other.u)),
                          *(self.u[i] * other.w + other.u[i] * self.w +
                            self.u[i - 2] * other.u[i - 1] - self.u[i - 1] * other.u[i - 2] for i in range(3)))

    def __invert__(self):
        q = sum(x ** 2 for x in self.u) + self.w ** 2
        return type(self)(self.w / q, *(-x / q for x in self.u))

    @classmethod
    def __from_axis_angle(cls, axis, angle):
        return cls(np.cos(angle / 2), *(np.array(axis) * np.sin(angle / 2)))

    @classmethod
    def x(cls, angle):
        axis = [1, 0, 0]
        return cls.__from_axis_angle(axis, angle)

    @classmethod
    def y(cls, angle):
        axis = [0, 1, 0]
        return cls.__from_axis_angle(axis, angle)

    @classmethod
    def z(cls, angle):
        axis = [0, 0, 1]
        return cls.__from_axis_angle(axis, angle)

    def __call__(self, r, alpha=1):
        return tuple((self * type(self)(0, *r) * (~self)).u), alpha

    def __add__(self, other):
        if isinstance(other, type(self)):
            return other * self
        else:
            return super().__add__(other)


class Expand(Transform):

    def __init__(self, inv=False, thr=0):
        self.inv = inv
        self.thr = thr

    def __repr__(self):
        return f'Expand(inv:{self.inv}, thr:{self.thr})'

    def __call__(self, r, alpha=1):
        if not self.inv:
            x, y = r
            with np.errstate(invalid='ignore'):
                z = np.sqrt(1 - x ** 2 - y ** 2)
            return (x, y, z), alpha
        else:
            x, y, z = r
            if isinstance(z, np.ndarray):
                t = np.where(z < self.thr)
                x[t] = np.nan
                y[t] = np.nan
            else:
                if z < self.thr:
                    x = np.nan
                    y = np.nan
            return (x, y), alpha

    def __invert__(self):
        return type(self)(not self.inv, self.thr)


class ToSpherical(Transform):

    def __init__(self, inv=False):
        self.inv = inv

    def __call__(self, r, alpha=1):
        if not self.inv:
            x, y, z = r
            theta = np.arcsin(x) * 180 / np.pi
            phi = np.arctan2(y, z) * 180 / np.pi
            phi = phi % 360

            return (theta, phi), alpha
        else:
            theta, phi = r
            x = np.sin(theta * np.pi / 180)
            y, z = (np.cos(theta * np.pi / 180) * np.sin(phi * np.pi / 180),
                    np.cos(theta * np.pi / 180) * np.cos(phi * np.pi / 180))
            return (x, y, z), alpha

    def __repr__(self):
        return f'ToSpherical(inv:{self.inv})'

    def __invert__(self):
        return ToSpherical(not self.inv)


class ToSynoptic(Transform):

    def __new__(cls, *args, **kwargs):
        if len(kwargs) == 5 and all(value == 0 for value in kwargs.values()):
            return Pipe()
        else:
            return super().__new__(cls)

    def __init__(self, crln, A=14.712, B=-2.396, C=-1.787, Wsid=14.184, Wsyn=360 / 27.2753):
        self.crln = crln
        self.A = A
        self.B = B
        self.C = C
        self.Wsid = Wsid
        self.Wsyn = Wsyn

    def __repr__(self):
        return f'ToSynoptic(A:{self.A}, B:{self.B}, C:{self.C})'

    def __call__(self, r, alpha=1):
        theta, phi = r
        sin_theta = np.sin(theta * np.pi / 180)

        dW = self.A - self.Wsid + self.B * sin_theta ** 2 + self.C * sin_theta ** 4
        dt = ((phi - self.crln - 180) % 360 - 180) / (self.Wsyn - dW)
        dphi = dW * dt
        return (theta, phi + dphi), alpha

    def __invert__(self):
        return type(self)(self.crln, A=self.A, B=self.B, C=self.C, Wsid=self.Wsid, Wsyn=-self.Wsyn)


class Filter(Transform):

    def __init__(self, func, inv=False):
        self.func = func
        self.inv = inv

    def __repr__(self):
        return f'Filter(func:{self.func}, inv:{self.inv})'

    def __call__(self, r, alpha=1):
        if not self.inv:
            return r, alpha * self.func(r)
        else:
            return r, alpha / self.func(r)

    def __invert__(self):
        return type(self)(self.func, not self.inv)


class Custom(Transform):

    def __init__(self, func, func_inv, **kwargs):
        self.func = func
        self.func_inv = func_inv
        self.params = kwargs

    def __repr__(self):
        return f'Custom({self.func.__repr__()}, {self.func_inv.__repr__()})'

    def __call__(self, r, alpha=1):
        return self.func(r, alpha, **self.params)

    def __invert__(self):
        return type(self)(self.func_inv, self.func, **self.params)


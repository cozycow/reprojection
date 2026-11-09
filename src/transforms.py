import numpy as np
from abc import ABC, abstractmethod


class Transform(ABC):

    @abstractmethod
    def __call__(self, *args):
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

    def __init__(self, *items, **kwargs):
        super().__init__(items)

    def __call__(self, *args):
        for transform in self:
            args = transform(*args)
        return args

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


class Normalize(Transform):
    def __new__(cls, *args, **kwargs):
        if all(abs(x) == 0. for x in args[0]) and all(x == 1. for x in args[1]):
            return Pipe()
        else:
            return super().__new__(cls)

    def __init__(self, shift, scale):
        self.shift = shift
        self.scale = scale

    def __repr__(self):
        return f'Normalize(shift:{self.shift}, scale:{self.scale})'

    def __call__(self, *args):
        return (tuple((x - a) / b for x, a, b in zip(args[:len(self.shift)], self.shift, self.scale)) +
                args[len(self.shift):])

    def __invert__(self):
        return type(self)(tuple(-x / y for x, y in zip(self.shift, self.scale)), tuple(1 / x for x in self.scale))

    def __add__(self, other):
        if isinstance(other, type(self)):
            return type(self)(tuple(x + y * a for x, y, a in zip(self.shift, other.shift, self.scale)),
                              tuple(a * b for a, b in zip(self.scale, other.scale)))
        else:
            return super().__add__(other)


class Rotate(Transform):

    def __new__(cls, *args, **kwargs):
        if np.all(np.array(args[0]) == 1.):
            return Pipe()
        else:
            return super().__new__(cls)


    def __init__(self, *args):
        self.w = args[0]
        self.u = args[1:]

    @property
    def axis(self):
        q = np.sqrt(sum(x ** 2 for x in self.u))
        return tuple(x / q for x in self.u)

    @property
    def angle(self):
        q = np.sqrt(sum(x ** 2 for x in self.u))
        return 2 * np.arctan2(q, self.w) * 180 / np.pi

    def __repr__(self):
        return f'Rotate(axis:, angle:)'

    def __mul__(self, other):
        return type(self)(self.w * other.w - sum(x * y for x, y in zip(self.u, other.u)),
                          *(self.u[i] * other.w + other.u[i] * self.w +
                            self.u[i - 2] * other.u[i - 1] - self.u[i - 1] * other.u[i - 2] for i in range(3)))

    def __invert__(self):
        q = sum(x ** 2 for x in self.u) + self.w ** 2
        return type(self)(self.w / q, *(-x / q for x in self.u))

    @classmethod
    def from_axis_angle(cls, axis, angle):
        axis = np.array(axis).astype(np.float32)
        axis /= np.linalg.norm(axis, axis=0) + 1e-16
        angle = np.expand_dims(np.array(angle), 0).astype(np.float32)

        while len(axis.shape) < len(angle.shape):
            axis = np.expand_dims(axis, -1)

        return cls(np.cos(angle / 2 * np.pi / 180), *(axis * np.sin(angle / 2 * np.pi / 180)))

    @classmethod
    def x(cls, angle):
        axis = [1, 0, 0]
        return cls.from_axis_angle(axis, angle)

    @classmethod
    def y(cls, angle):
        axis = [0, 1, 0]
        return cls.from_axis_angle(axis, angle)

    @classmethod
    def z(cls, angle):
        axis = [0, 0, 1]
        return cls.from_axis_angle(axis, angle)

    def __call__(self, *args):
        return tuple(np.squeeze((self * type(self)(0, *args[:3]) * (~self)).u)) + args[3:]

    def __add__(self, other):
        if isinstance(other, type(self)):
            return other * self
        else:
            return super().__add__(other)


class DiffRotate(Transform):

    def __new__(cls, *args, **kwargs):
        if args[0] == 0.:
            return Pipe()
        else:
            return super().__new__(cls)

    def __init__(self, func, time=1, **kwargs):
        self.func = func
        self.time = time
        self.kwargs = kwargs

    def __repr__(self):
        return f'CorrectDR(jdn:{self.time})'

    def __call__(self, *args):
        W = self.func(args[:3], **self.kwargs)
        rot = Rotate.y(W * self.time)
        return rot(*args)

    def __invert__(self):
        return type(self)(self.func, -self.time, **self.kwargs)

    def __add__(self, other):
        if (isinstance(other, type(self)) and (self.func == other.func) and
            all(self.kwargs[key] == other.kwargs[key] for key in self.kwargs.keys())):
            return type(self)(self.time + other.time, self.func, **self.kwargs)
        else:
            return super().__add__(other)


class To3d(Transform):

    def __init__(self, inv=False):
        self.inv = inv

    def __repr__(self):
        return f'Make3d(inv:{self.inv})'

    def __call__(self, *args):
        if not self.inv:
            y, x = args[:2]
            z = 1 - x ** 2 - y ** 2

            t = np.where(z < 0)
            x[t], y[t], z[t] = np.nan, np.nan, np.nan
            z = np.sqrt(z)
            return (x, y, z) + args[2:]
        else:
            x, y, z = args[:3]

            t = np.where(z < -1e-8)
            x[t], y[t] = np.nan, np.nan
            return (y, x) + args[3:]

    def __invert__(self):
        return type(self)(not self.inv)


class ToParaxial(Transform):

    def __new__(cls, *args, **kwargs):
        if args[0] == 0.:
            return Pipe()
        else:
            return super().__new__(cls)

    def __init__(self, theta):
        self.theta = theta

    def __repr__(self):
        return f'ToParaxial(theta:{self.theta})'

    def __call__(self, *args):
        x, y, z = args[:3]

        q = np.sin(self.theta * np.pi / 180)
        p = 1 / (1 + q * z)

        u = np.sqrt(1 - q ** 2) * p
        x, y = x * u, y * u
        z = (z + q) * p
        return (x, y, z) + args[3:]

    def __invert__(self):
        return type(self)(-self.theta)

    def __add__(self, other):
        if isinstance(other, type(self)):
            q1 = np.sin(self.theta * np.pi / 180)
            q2 = np.sin(other.theta * np.pi / 180)
            theta = np.arcsin((q1 + q2) / (1 + q1 * q2)) * 180 / np.pi
            return type(self)(theta)
        else:
            return super().__add__(other)


class MuFilter(Transform):

    def __init__(self, minmu=0., theta=0., pow=1.):
        self.minmu = minmu
        self.theta = theta
        self.pow = pow

    def __repr__(self):
        return f'MuFilter(pow:{self.pow})'

    def __call__(self, *args):
        z = args[2]
        q = np.sin(self.theta * np.pi / 180)
        mu = (z - q) / np.sqrt(q ** 2 - 2 * q * z + 1)
        t = np.where(mu < self.minmu)
        mu[t] = np.nan

        return args[:-1] + (args[-1] * mu ** self.pow,)

    def __invert__(self):
        return type(self)(self.minmu, self.theta, -self.pow)


class ToSpherical(Transform):

    def __init__(self, sine=False, inv=False):
        self.sine = sine
        self.inv = inv

    def __call__(self, *args):
        if not self.inv:
            x, y, z = args[:3]
            if self.sine:
                theta = y * 90
            else:
                theta = np.arcsin(y) * 180 / np.pi
            phi = np.arctan2(x, z) * 180 / np.pi % 360
            return (theta, phi) + args[3:]
        else:
            theta, phi = args[:2]
            if self.sine:
                y = theta / 90
            else:
                y = np.sin(theta * np.pi / 180)
            a = np.sqrt(1 - y ** 2)
            x, z = (a * np.sin(phi * np.pi / 180),
                    a * np.cos(phi * np.pi / 180))
            return (x, y, z) + args[2:]

    def __repr__(self):
        return f'ToSpherical(sine:{self.sine}, inv:{self.inv})'

    def __invert__(self):
        return ToSpherical(self.sine, not self.inv)


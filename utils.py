import numpy as np
import torch
from numbers import Number


def to_tensor(x, cuda_default=True):
    if isinstance(x, np.ndarray):
        # pytorch doesn't support bool
        if x.dtype == "bool":
            x = x.astype("int")
        # we want only single precision floats
        if x.dtype == "float64":
            x = x.astype("float32")

        x = torch.from_numpy(x)

    if isinstance(x, torch.Tensor) and cuda_default and torch.cuda.is_available():
        x = x.cuda()

    return x


def to_np(value):
    if isinstance(value, Number):
        return np.array(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    # If iterable
    try:
        return np.array([to_np(v) for v in value])
    except TypeError:
        return np.array(value)

    raise ValueError("Data type {} not supported".format(value.__class__.__name__))


def discounted_sum_rewards(rewards, dones, last_state_value_t=None, gamma=0.99):
    # Expected shape = (num_samples, num_envs)
    rewards = rewards.copy()

    if last_state_value_t is not None:
        bootstrap = np.where(dones[-1] == False)[0]
        rewards[-1][bootstrap] = last_state_value_t[bootstrap]

    returns = np.zeros(rewards.shape)
    returns_sum = np.zeros(rewards.shape[-1])

    for i in reversed(range(rewards.shape[0])):
        returns_sum = rewards[i] + gamma * returns_sum * (1 - dones[i])
        returns[i] = returns_sum

    return returns


class SimpleMemory(dict):
    """
    A dict whose keys can be accessed as attributes.

    Parameters
    ----------
    initial_keys: list of strings
        Each key will be initialized as an empty list.
    """

    def __init__(self, *args, initial_keys=None, **kwargs):
        super().__init__(*args, **kwargs)

        initial_keys = initial_keys or []
        for k in initial_keys:
            self[k] = []

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)

    @classmethod
    def from_list_of_dicts(cls, dicts):
        return cls({k: [d[k] for d in dicts] for k in dicts[0]})


class MeanStdFilter:
    """
    Calculates the exact mean and std deviation, originally by `ray_rllib
    <https://goo.gl/fMv49b>`_.

    Parameters
    ----------
    shape: tuple
        The shape of the inputs to :meth:`self.normalize` and :meth:`self.scale`
    clip_range: float
        The output of :meth:`self.normalize` and :meth:`self.scale` will be clipped
        to this range, use np.inf for no clipping. (Default is 5)
    """

    def __init__(self, num_features, clip_range=5.):
        if not isinstance(num_features, int):
            raise ValueError(
                "num_features should be an int, got {}".format(num_features)
            )

        self.num_features = num_features
        self.clip_range = clip_range
        self.n = 0
        self.xs = []

        self.M = np.zeros(num_features)
        self.S = np.zeros(num_features)

    def _check_shape(self, x):
        if not (self.num_features,) == x.shape[1:]:
            raise ValueError(
                "Data shape must be (num_samples, {}) but is {}".format(
                    self.num_features, x.shape
                )
            )

    @property
    def mean(self):
        return self.M

    @property
    def var(self):
        if self.n == 0 or self.n == 1:
            return np.ones(self.S.shape)
        else:
            return self.S / (self.n - 1)

    @property
    def std(self):
        return np.sqrt(self.var)

    def update(self):
        n_old = self.n
        n_new = len(self.xs)
        if n_new == 0:
            return

        x = to_np(self.xs)
        self.n += n_new
        self.xs = []

        x_mean = x.mean(axis=0)
        x_std = ((x - x_mean) ** 2).sum(axis=0)
        # First update
        if self.n == n_new:
            self.M[:] = x_mean
            self.S[:] = x_std
        else:
            new_mean = (n_old * self.M + n_new * x_mean) / self.n
            self.S[:] = self.S + x_std + (self.M - x_mean) ** 2 * n_old * n_new / self.n
            self.M[:] = new_mean

    def normalize(self, x, add_sample=True):
        """
        Normalizes x by subtracting the mean and dividing by the standard deviation.

        Parameters
        ----------
        add_sample: bool
            If True x will be added as a new sample and will be considered when
            the filter is updated via :meth:`self.update`. (Default is True)
        """
        self._check_shape(x)
        if add_sample:
            self.xs.extend(x)

        return (x - self.mean) / (self.std + 1e-7)

    def scale(self, x, add_sample=True):
        """
        Scales x by dividing by the standard deviation.

        Parameters
        ----------
        add_sample: bool
            If True x will be added as a new sample and will be considered when
            the filter is updated via :meth:`self.update`. (Default is True)
        """
        self._check_shape(x)
        if add_sample:
            self.xs.extend(x)

        return x / (self.std + 1e-7)

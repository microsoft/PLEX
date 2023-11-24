import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from robomimic.models.distributions import TanhWrappedDistribution


class DiagonalGaussian(D.Distribution):
    def __init__(self, loc, scale):
        assert loc.shape == scale.shape
        assert (scale > 0).all()
        self.loc = loc
        self.scale = scale
        self._batch_shape = loc.shape[:-1]
        self._event_shape = loc.shape[-1:]

        self._impl = D.Independent(D.Normal(loc, scale), 1)
        assert self._impl.batch_shape == self._batch_shape
        assert self._impl.event_shape == self._event_shape

        for method in ['sample', 'rsample', 'log_prob']:
            setattr(self, method, getattr(self._impl, method))


# Borrowed from https://github.com/denisyarats/pytorch_sac
class TanhTransform(D.transforms.Transform):
    domain = D.constraints.real
    codomain = D.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedGaussian(D.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        super().__init__(DiagonalGaussian(loc, scale), TanhTransform())

    @property
    def mean(self):
        mu = self.base_dist.loc
        for transform in self.transforms:
            mu = transform(mu)
        return mu


def _head(in_dim, out_dim, hidden_dim=None):
    if hidden_dim is None:
        return nn.Linear(in_dim, out_dim)
    else:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

def _rescale(x, min, max):
    return min + (max - min) * torch.sigmoid(x)
    

class GaussianHead(nn.Module):
    def __init__(self, input_dim, output_dim, std_bounds,
                 hidden_dim=None, squash=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_bounds = std_bounds
        self.squash = squash
        self.mean_head = _head(input_dim, output_dim, hidden_dim=hidden_dim)
        self.std_head = _head(input_dim, output_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        mean = self.mean_head(x)
        std = _rescale(self.std_head(x), *self.std_bounds)
        dist = D.Normal(loc=mean, scale=std)
        # dist = D.Independent(dist, 1)   # diagonal

        if self.squash:
            dist = TanhWrappedDistribution(dist)

        return dist


class GaussianMixtureHead(nn.Module):
    def __init__(self, num_components, input_dim, output_dim, std_bounds,
                 hidden_dim=None, squash=False):
        super().__init__()
        self.num_components = num_components
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.squash = squash
        self.std_bounds = std_bounds
        self.mean_heads = nn.ModuleList([
            _head(input_dim, output_dim, hidden_dim=hidden_dim)
            for _ in range(num_components)
        ])
        self.std_heads = nn.ModuleList([
            _head(input_dim, output_dim, hidden_dim=hidden_dim)
            for _ in range(num_components)
        ])
        self.logits_head = _head(input_dim, num_components, hidden_dim=hidden_dim)

    def forward(self, x):
        # mixture dim will come right after other batch dims
        batch_shape = tuple(x.shape[:-1])
        mixture_dim = len(batch_shape)

        # unnormalized logits to categorical distribution for mixing the modes
        logits = self.logits_head(x)
        mixture = D.Categorical(logits=logits)

        means = torch.stack([head(x) for head in self.mean_heads], dim=mixture_dim)
        stds = _rescale(
            torch.stack([head(x) for head in self.std_heads], dim=mixture_dim),
            *self.std_bounds
        )
        dists = D.Normal(loc=means, scale=stds)
        dists = D.Independent(dists, 1)     # diagonal
        dist = D.MixtureSameFamily(mixture_distribution=mixture, component_distribution=dists)

        if self.squash:
            dist = TanhWrappedDistribution(dist)

        return dist


def slice_dist(dist, key):
    """ Recursively slicing a Distribution object. """
    batch_shape = dist.batch_shape
    new = type(dist).__new__(type(dist))
    for k, v in dist.__dict__.items():
        if isinstance(v, D.Distribution):
            sliced_v = slice_dist(v, key)
        elif isinstance(v, torch.Tensor) and batch_shape == v.shape[:len(batch_shape)]:
            sliced_v = v[key]
        elif 'batch_shape' in k:
            device = dist.loc.device
            sliced_v = torch.zeros(v, device=device)[key].shape
        else:
            sliced_v = v
        setattr(new, k, sliced_v)
    return new
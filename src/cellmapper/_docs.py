"""Shared documentation for cellmapper."""

from docrep import DocstringProcessor

__all__ = ["d"]

_t = """\
t
    Number of diffusion time steps. This parameter controls the degree of
    smoothing applied by the diffusion operator. Larger values lead to more
    smoothing."""

_diffusion_method = """\
diffusion_method
    Method for computing the diffusion operator (only valid in self-mapping mode). Options are "iterative" for
    iterative matrix multiplication (inspired by MAGIC :cite:`van2018recovering`) and "spectral" for
    eigendecomposition-based approach. """

_prediction_postfix = """\
prediction_postfix
    Postfix to add to mapped variables to identify them as predictions."""

d = DocstringProcessor(
    t=_t,
    diffusion_method=_diffusion_method,
    prediction_postfix=_prediction_postfix,
)

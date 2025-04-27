"""Dependency checking."""

import importlib
import types

from packaging.version import parse

from . import version


class Checker:
    """
    Checks availability and version of a Python module dependency.

    Adapted from the scGLUE package: https://github.com/gao-lab/GLUE

    Parameters
    ----------
    name
        Name of the dependency
    package_name
        Name of the package to check version for (if different from module name)
    vmin
        Minimal required version
    install_hint
        Install hint message to be printed if dependency is unavailable
    """

    def __init__(
        self, name: str, package_name: str | None = None, vmin: str | None = None, install_hint: str | None = None
    ) -> None:
        self.name = name
        self.package_name = package_name or name
        self.vmin = parse(vmin) if vmin else vmin
        vreq = f" (>={self.vmin})" if self.vmin else ""
        self.vreq_hint = f"This function relies on {self.name}{vreq}."
        self.install_hint = install_hint

    def check(self) -> None:
        """Check if the dependency is available and meets the version requirement."""
        try:
            importlib.import_module(self.name)
        except ModuleNotFoundError as e:
            raise RuntimeError(" ".join(filter(None, [self.vreq_hint, self.install_hint]))) from e
        v = parse(version(self.package_name))
        if self.vmin and v < self.vmin:
            raise RuntimeError(
                " ".join(
                    [
                        self.vreq_hint,
                        f"Detected version is {v}.",
                        "Please install a newer version.",
                        self.install_hint or "",
                    ]
                )
            )


INSTALL_HINTS = types.SimpleNamespace(
    rapids="To speed up k-NN search on GPU, you may install rapids following the guide from "
    "https://docs.rapids.ai/install/. Note that you will only need cuML.",
    faiss="To speed up k-NN search on GPU, you may install faiss following the guide from "
    "https://github.com/facebookresearch/faiss/blob/main/INSTALL.md",
)

CHECKERS = {
    "rapids": Checker("rapids", vmin=None, install_hint=INSTALL_HINTS.rapids),
    "faiss": Checker("faiss", package_name="faiss", vmin="1.7.0", install_hint=INSTALL_HINTS.faiss),
}


def check_deps(*args) -> None:
    """
    Check whether certain dependencies are installed

    Parameters
    ----------
    args
        A list of dependencies to check
    """
    for item in args:
        CHECKERS[item].check()

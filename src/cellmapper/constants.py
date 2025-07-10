class PackageConstants:
    """Constants used througout the package."""

    N_COMPS: int = 50
    # Cutoff for using sklearn neighbor search; above this, warn user
    SKLEARN_WARNING_CUTOFF: int = 50000

    # Default mapping methods
    DEFAULT_SELF_MAPPING_METHOD: str = "umap"
    DEFAULT_CROSS_MAPPING_METHOD: str = "gauss"

    # Kernel methods that only work in self-mapping mode
    SELF_MAPPING_ONLY_KERNELS = {"umap", "adaptive_gauss"}

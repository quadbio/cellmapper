class PackageConstants:
    """Constants used througout the package."""

    N_COMPS: int = 50
    # Cutoff for using sklearn neighbor search; above this, warn user
    SKLEARN_WARNING_CUTOFF: int = 50000

    # Default mapping methods
    DEFAULT_SELF_MAPPING_KERNEL_METHOD: str = "umap"
    DEFAULT_CROSS_MAPPING_KERNEL_METHOD: str = "gauss"

    # Kernel method categories
    JACCARD_BASED_KERNELS = {"jaccard", "hnoca"}
    CONNECTIVITY_BASED_KERNELS = {"gauss", "scarches", "inverse_distance", "random", "equal", "umap"}

    # Kernel methods that only work in self-mapping mode
    SELF_MAPPING_ONLY_KERNELS = {"umap"}

    # Threshold for recommending spectral method over iterative for matrix powers
    SPECTRAL_METHOD_THRESHOLD: int = 10

"""Helper macros for CHEDDAR opt-in build configuration."""

def requires_cheddar():
    """Returns target_compatible_with for CHEDDAR-requiring targets."""
    return select({
        "@heir//:config_enable_cheddar": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

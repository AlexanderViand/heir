"""Build-time gating helpers for FIDESlib-backed targets."""

def if_fideslib_enabled(if_true, if_false = None):
    """Select between values based on --//:enable_fideslib."""
    if if_false == None:
        if_false = []
    return select({
        "@heir//:config_enable_fideslib": if_true,
        "//conditions:default": if_false,
    })

def requires_fideslib():
    """Marks a target as incompatible unless --//:enable_fideslib=1."""
    return select({
        "@heir//:config_enable_fideslib": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def fideslib_deps(extra = None):
    """Returns deps including @fideslib//:fideslib only when enabled."""
    if extra == None:
        extra = []
    return if_fideslib_enabled(
        ["@fideslib//:fideslib"] + extra,
        [],
    )

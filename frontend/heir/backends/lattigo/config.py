"""Configuration for the Lattigo backend."""

import os
import os.path
from heir.backends.util.common import get_repo_root


try:
  # Python 3.9+
  from dataclasses import dataclass
except ImportError:
  from dataclasses import dataclass


@dataclass(frozen=True)
class LattigoConfig:
  """Configuration for the Lattigo backend.

  Attributes:
      pkg_dir: the root directory of the Lattigo Go module.
  """

  pkg_dir: str


def from_os_env(debug: bool = False) -> LattigoConfig:
  """Create a LattigoConfig from environment variables.

  Environment variables:
    - LATTIGO_PKG_DIR: path to the root of the Lattigo Go module.

  Args:
      debug: whether to print debug information.

  Returns:
      A LattigoConfig pointing at the Go module for Lattigo.
  """
  pkg_dir = os.environ.get("LATTIGO_PKG_DIR", "")
  if debug:
    print(f"LATTIGO_PKG_DIR={pkg_dir}")
  # Support Bazel runfiles layout
  if ("RUNFILES_DIR" in os.environ or "TEST_SRCDIR" in os.environ) and pkg_dir:
    base = os.getenv("RUNFILES_DIR", os.getenv("TEST_SRCDIR", ""))
    pkg_dir = os.path.join(base, pkg_dir)
  if not pkg_dir:
    raise RuntimeError(
        "LATTIGO_PKG_DIR environment variable must be set to the Lattigo Go"
        " module root"
    )
  return LattigoConfig(pkg_dir=pkg_dir)

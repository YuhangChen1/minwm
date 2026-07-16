from __future__ import annotations

from pathlib import Path

from Full_Duplex_Fix.config import load_config
from Full_Duplex_Fix.preencode import _validate_manifest


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "overfit.yaml"
    config = load_config(config_path)
    result = _validate_manifest(config)
    print(result)


if __name__ == "__main__":
    main()

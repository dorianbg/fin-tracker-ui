from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from raa_official import fetch_official_raa_allocation  # noqa: E402


if __name__ == "__main__":
    allocation = fetch_official_raa_allocation(use_cache=False)
    print(f"Downloaded RAA allocation as of {allocation.as_of} from {allocation.source}")
    for asset, weight in sorted(allocation.allocation.items()):
        print(f"{asset}: {weight:.2%}")

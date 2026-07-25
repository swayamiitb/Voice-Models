"""Thin entry point so `vajravoice-synthesize` works after `pip install`."""

from scripts.synthesize import main

if __name__ == "__main__":
    raise SystemExit(main())

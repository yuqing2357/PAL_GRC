#!/usr/bin/env python3
"""Launch one D3--D4 bridge-controlled capacity run."""
from __future__ import annotations

import argparse

from ssia_alignment.config import load_config
from ssia_alignment.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(load_config(args.config, args.set), resume_override=args.resume)


if __name__ == "__main__":
    main()

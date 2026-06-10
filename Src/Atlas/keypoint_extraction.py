import runpy

from Src.Atlas.inference.extraction import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("Src.Atlas.inference.extraction", run_name="__main__")

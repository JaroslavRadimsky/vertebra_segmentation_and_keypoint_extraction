import runpy

from Src.Atlas.inference.inference import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("Src.Atlas.inference.inference", run_name="__main__")

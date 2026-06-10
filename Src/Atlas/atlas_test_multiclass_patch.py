import runpy

from Src.Atlas.evaluation.atlas_test_multiclass_patch import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("Src.Atlas.evaluation.atlas_test_multiclass_patch", run_name="__main__")

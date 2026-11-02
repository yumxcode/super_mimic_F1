"""Compatibility shims the Gradmotion entry applies inside the training process, before allmimic is imported.

Only one shim so far, and it is gated twice (env var + driver check) so it cannot silently change a run that does
not need it.

`mujoco_warp.put_model` calls `warp_util.check_toolkit_driver()`, which raises `RuntimeError("Minimum supported CUDA
version: 12.4.")` when `warp.is_conditional_graph_supported()` is false. On the Gradmotion 4090D hosts the driver is
535.183.01 (= CUDA 12.2), so that predicate is false and every mjlab run dies in `Simulation.__init__` -- even though

  * mjlab itself handles the same limit gracefully: `mjlab/sim/sim.py:_should_use_cuda_graph()` returns False when the
    driver is older than 12.4 and prints "CUDA Graphs disabled: driver 12.2 < 12.4", i.e. mjlab is designed to run
    without CUDA graphs on this driver; and
  * conditional graphs are what the check is about, and mujoco-warp 3.5.0 does not use them anywhere: the string
    "conditional_graph" appears exactly once in the whole package, in that guard.

So the shim replaces the guard with a logged no-op instead of lying about a capability: nothing that the guard
protects is reachable in this version, and the actual consequence (no CUDA graphs) is decided by mjlab's own check,
which reports itself in the log.

  GM_ALLOW_OLD_CUDA_DRIVER=1 python -c "import sys; sys.path.insert(0, 'scripts'); import gm_compat; gm_compat.apply(); \
      import runpy; sys.argv = ['allmimic.train', ...]; runpy.run_module('allmimic.train', run_name='__main__')"
"""

from __future__ import annotations

import os


def apply() -> bool:
    """Patch mujoco_warp's driver gate. Returns True if the patch was installed."""
    if os.environ.get('GM_ALLOW_OLD_CUDA_DRIVER') != '1':
        return False
    try:
        import warp as wp
        from mujoco_warp._src import warp_util
    except Exception as e:  # noqa: BLE001
        print(f'[gm-compat] not applied ({type(e).__name__}: {e})', flush=True)
        return False
    wp.init()
    drv = wp.context.runtime.driver_version
    if wp.is_conditional_graph_supported():
        print(f'[gm-compat] driver {drv} supports conditional graphs; mujoco_warp gate left as is', flush=True)
        return False

    def check_toolkit_driver() -> None:
        print(
            f'[gm-compat] driver {drv.minor if hasattr(drv, "minor") else drv}: mujoco_warp 3.5.0 refuses to run below '
            f'driver 12.4 because of conditional CUDA graphs; the gate is bypassed because mjlab disables CUDA graphs '
            f'on this driver (mjlab/sim/sim.py:_should_use_cuda_graph) and mujoco-warp 3.5.0 uses no conditional '
            f'graph anywhere else',
            flush=True,
        )

    warp_util.check_toolkit_driver = check_toolkit_driver
    print(f'[gm-compat] patched mujoco_warp._src.warp_util.check_toolkit_driver (driver {drv})', flush=True)
    return True

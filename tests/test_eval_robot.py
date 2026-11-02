"""Regression check: every eval env is built for the checkpoint's robot, never the `g1` default.

`make_eval_env(..., robot='g1')` is the signature default, so a call site that forgets the argument silently evaluates
an X1 policy inside a G1 simulator. That happened: TASK_20261102_043 trained the X1 for 3000 iterations and its
end-of-run evaluation reported "0/41" computed in the G1 env, which the report then quoted as an X1 benchmark.

This test is static (no GPU, no mjlab): it parses the call sites and requires an explicit robot argument, and it
additionally runs the assertion in Trainer.run_eval's source. A runtime counterpart lives in
allmimic/train.py (`assert self.eval_env.robot_cfg.name == cfg.robot`), which catches the same mistake on the platform
where the env can actually be constructed.

  pytest tests/test_eval_robot.py
  PYTHONPATH=. python tests/test_eval_robot.py
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# every module that builds an evaluation environment
CALL_SITES = (
    'allmimic/train.py',
    'allmimic/adapt.py',
    'allmimic/ppo.py',
    'allmimic/evaluate.py',
    'scripts/push_test.py',
    'tools/eval_allmimic.py',
    'tools/wm_grad_fidelity.py',
)


def _robot_arg(node: ast.Call) -> ast.expr | None:
    """the robot argument, whether passed by keyword or as the 5th positional (lib, n, device, tcfg, robot)"""
    for kw in node.keywords:
        if kw.arg == 'robot':
            return kw.value
    return node.args[4] if len(node.args) >= 5 else None


def check_calls() -> list[str]:
    problems = []
    for rel in CALL_SITES:
        path = ROOT / rel
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'make_eval_env'):
                continue
            robot = _robot_arg(node)
            if robot is None:
                problems.append(f'{rel}:{node.lineno} calls make_eval_env without a robot argument '
                                f'(defaults to g1)')
            elif isinstance(robot, ast.Constant) and robot.value == 'g1':
                problems.append(f'{rel}:{node.lineno} hardcodes robot="g1"')
    return problems


def check_trainer_assertion() -> list[str]:
    """Trainer.run_eval must assert the eval env's robot matches the training robot."""
    src = (ROOT / 'allmimic/train.py').read_text()
    problems = []
    if 'eval_env.robot_cfg.name == cfg.robot' not in src:
        problems.append('allmimic/train.py: run_eval does not assert eval_env.robot_cfg.name == cfg.robot')
    if 'robot=cfg.robot' not in src:
        problems.append('allmimic/train.py: run_eval does not pass robot=cfg.robot to make_eval_env')
    return problems


def main() -> int:
    problems = check_calls() + check_trainer_assertion()
    if problems:
        print('FAIL')
        for p in problems:
            print('  -', p)
        return 1
    print(f'OK: {len(CALL_SITES)} modules build eval envs with an explicit robot, and run_eval asserts it')
    return 0


def test_eval_envs_use_the_checkpoint_robot() -> None:
    problems = check_calls()
    assert not problems, '\n'.join(problems)


def test_run_eval_asserts_the_robot() -> None:
    problems = check_trainer_assertion()
    assert not problems, '\n'.join(problems)


if __name__ == '__main__':
    raise SystemExit(main())

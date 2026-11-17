"""Run unchanged RCWA-FFF with an adjustable geometry-plan memory budget."""
from pathlib import Path
import argparse
import sys
import time

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--plan-cache-mib', type=int, default=1024)
    args, extra = p.parse_known_args(argv)
    if args.plan_cache_mib < 0:
        p.error('--plan-cache-mib must be nonnegative.')
    from model import solver_root, save_json
    from fff1d import run_fff
    child = run_fff.parser().parse_args(extra)
    solver_root(child.solver_root)
    import fast_solver
    from fff1d import operators
    fast_solver._MAX_BYTES = args.plan_cache_mib * 1024 * 1024
    operators._LIMIT = args.plan_cache_mib * 1024 * 1024
    original = run_fff.calculate
    fresh_points = 0

    def timed(*positional, **keywords):
        nonlocal fresh_points
        started = time.perf_counter()
        result = original(*positional, **keywords)
        fresh_points += 1
        row = result['summary'][0]
        print(f"[timing] P={row['period_nm']:g} {row['void']} {row['wavelength_nm']:g} nm: "
              f"fresh_point_seconds={time.perf_counter()-started:.3f}", flush=True)
        return result

    run_fff.calculate = timed
    started = time.perf_counter()
    try:
        return run_fff.main(extra)
    finally:
        run_fff.calculate = original
        if child.output_dir and child.output_dir.is_dir():
            save_json(child.output_dir / 'performance.json', dict(
                elapsed_seconds=time.perf_counter()-started,
                fresh_points=fresh_points, plan_cache_mib=args.plan_cache_mib,
                interval_plan_cache=fast_solver.plan_cache_info(),
                fff_plan_cache=operators.cache_info(),
                note='Only memory retention and timing display differ; physics modules/cache keys unchanged.'))


if __name__ == '__main__':
    raise SystemExit(main())

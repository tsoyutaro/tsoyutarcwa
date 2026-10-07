"""Preview the actual Au/PMMA/air staircase without importing RCWA or torch."""
from __future__ import annotations

if __package__:
    from .common import HERE, load_config, geometry_svg, parser, write_json
else:
    from common import HERE, load_config, geometry_svg, parser, write_json


def main():
    p = parser(__doc__)
    p.add_argument("--slices", type=int)
    args = p.parse_args()
    config, _ = load_config(args.config)
    slices = args.slices if args.slices is not None else config["fixed_numerics"]["slices"]
    output = args.output_dir or HERE/"results/preview"
    output.mkdir(parents=True, exist_ok=True)
    geometry_svg(config, output/"geometry.svg", slices)
    write_json(output/"settings.json", {"config": config, "profile_slices": slices,
                                       "total_finite_layers": slices+int(config["geometry"]["gold_cap_thickness_nm"] > 0)})
    print(f"structure: {output/'geometry.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

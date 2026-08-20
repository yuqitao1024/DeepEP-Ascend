import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.benchmark.report_markdown import (  # noqa: E402
    identify_profile,
    render_comparison_markdown,
    validate_complete_report,
    write_text_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", required=True, type=Path)
    parser.add_argument("--ascend", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _load_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("report")
    return report


def _identify_profiles(cuda: dict, ascend: dict):
    cuda_profile = None
    ascend_profile = None
    cuda_error = None
    ascend_error = None
    try:
        cuda_profile = identify_profile(cuda)
    except ValueError as error:
        cuda_error = error
    try:
        ascend_profile = identify_profile(ascend)
    except ValueError as error:
        ascend_error = error

    if cuda_profile is None:
        if ascend_profile is not None:
            validate_complete_report(
                cuda,
                platform="cuda",
                profile=ascend_profile,
                require_h800=True,
            )
        raise cuda_error
    if ascend_profile is None:
        validate_complete_report(
            ascend,
            platform="ascend",
            profile=cuda_profile,
        )
        raise ascend_error
    if cuda_profile.name != ascend_profile.name:
        raise ValueError("profile")
    return cuda_profile, ascend_profile


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cuda = _load_report(args.cuda)
        ascend = _load_report(args.ascend)
        cuda_profile, ascend_profile = _identify_profiles(cuda, ascend)
        validate_complete_report(
            cuda,
            platform="cuda",
            profile=cuda_profile,
            require_h800=True,
        )
        validate_complete_report(
            ascend,
            platform="ascend",
            profile=ascend_profile,
        )
        markdown = render_comparison_markdown(cuda, ascend, cuda_profile)
        write_text_atomic(args.output, markdown)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

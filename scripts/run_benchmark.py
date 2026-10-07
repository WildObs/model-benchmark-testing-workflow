"""Command line entry point for the model benchmark workflow.

Examples:
    python scripts/run_benchmark.py --list-templates
    python scripts/run_benchmark.py wildobs-national-20260428055250 --template minimal
    python scripts/run_benchmark.py my-export --exclude-sections Limitations Full_Confusion_Matrices
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_THRESHOLDS, BenchmarkConfig  # noqa: E402
from src.pipeline import run_benchmark  # noqa: E402
from src.templating import list_sections, list_templates, resolve_template  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an AI species recognition model using a Camtrap DP export.")
    parser.add_argument("input_folder", nargs="?", help="Name of the Camtrap DP folder inside the input path")
    parser.add_argument("--source-location", default="", help="Where the test images came from")
    parser.add_argument("--collation-process", default="", help="How the test dataset was collated")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS, help="Confidence thresholds")
    parser.add_argument("--no-export-errors", action="store_true", help="Skip the misclassified images CSV")
    parser.add_argument("--template", default="full", help="Template name or path to a .md template")
    parser.add_argument("--exclude-sections", nargs="*", default=[], help="Template sections to leave out")
    parser.add_argument("--input-path", default="Input_Data")
    parser.add_argument("--output-path", default="Output_Reports")
    parser.add_argument("--templates-path", default="templates")
    parser.add_argument("--log-path", default="logs")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--list-templates", action="store_true", help="List templates and their sections, then exit")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_templates:
        templates_path = Path(args.templates_path)
        if not templates_path.is_absolute():
            templates_path = PROJECT_ROOT / templates_path
        for name, description in list_templates(templates_path).items():
            sections = list_sections(resolve_template(name, templates_path))
            print(f"{name}: {description}\n    sections: {', '.join(sections)}")
        return 0

    if not args.input_folder:
        print("error: input_folder is required (or use --list-templates)", file=sys.stderr)
        return 2

    config = BenchmarkConfig(
        input_camtrap_folder_name=args.input_folder,
        data_source_location=args.source_location,
        data_collation_process=args.collation_process,
        thresholds=args.thresholds,
        export_errors=not args.no_export_errors,
        template=args.template,
        exclude_sections=args.exclude_sections,
        input_path=args.input_path,
        output_path=args.output_path,
        templates_path=args.templates_path,
        log_path=args.log_path,
        log_level=args.log_level,
    )
    run = run_benchmark(config)
    print(f"Report saved to: {run.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

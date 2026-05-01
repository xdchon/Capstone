from __future__ import annotations

import argparse
from pathlib import Path

import csv_openai
import nli_gui


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask OpenAI about a CSV file directly.")
    parser.add_argument("csv_path", help="CSV file to analyse directly")
    parser.add_argument("--question", "-q", default=None, help="Question to ask about the CSV")
    parser.add_argument("--report", action="store_true", help="Generate the OpenAI CSV report package")
    parser.add_argument("--out", default="openai_csv_cli_output", help="Output folder for --report")
    parser.add_argument("--model", default=nli_gui.DEFAULT_OPENAI_MODEL)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    profile = csv_openai.csv_profile(csv_path, preview_rows=0)
    print(f"CSV: {csv_path}")
    print(f"Rows: {profile['row_count']}")
    print(f"Columns: {len(profile['columns'])}")

    client = nli_gui.get_openai_client()

    if args.report:
        report = csv_openai.build_openai_csv_report_package(
            client=client,
            csv_path=csv_path,
            output_dir=Path(args.out),
            model=nli_gui.DEFAULT_REPORT_OPENAI_MODEL,
            reasoning_effort=nli_gui.DEFAULT_REPORT_REASONING_EFFORT,
        )
        print(f"Report written to: {report}")
        return

    question = args.question or input("Question: ").strip()
    if not question:
        raise RuntimeError("No question provided.")
    answer = csv_openai.answer_csv_question(client, csv_path, question, model=args.model)
    print(answer)


if __name__ == "__main__":
    main()

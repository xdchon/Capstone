from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st

import csv_openai
import nli_gui as core


BASE_DIR = Path(__file__).resolve().parent
CSV_PROJECT_DIR = BASE_DIR / "csv_projects" / "streamlit_project"
CSV_DATA_DIR = CSV_PROJECT_DIR / "source_csv_files"


def ensure_project() -> None:
    CSV_DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.set_active_csv_project(CSV_PROJECT_DIR)


def list_csvs() -> List[str]:
    ensure_project()
    return sorted(path.name for path in CSV_DATA_DIR.glob("*.csv") if path.is_file())


def csv_path(name: str) -> Path:
    path = CSV_DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return path


def main() -> None:
    ensure_project()
    st.set_page_config(page_title="NLI CSV OpenAI Chat", layout="wide")
    st.title("NLI CSV OpenAI Chat")
    st.caption("Upload CSV files, choose one CSV, then query or report on that CSV directly with OpenAI.")

    uploaded = st.file_uploader("Add CSV file", type=["csv"])
    if uploaded is not None and st.button("Add this CSV"):
        dest = CSV_DATA_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        st.success(f"Added {uploaded.name}")

    datasets = list_csvs()
    if not datasets:
        st.info("Add a CSV file to begin.")
        return

    selected_dataset = st.selectbox("CSV dataset", datasets)
    selected_path = csv_path(selected_dataset)
    profile = csv_openai.csv_profile(selected_path, preview_rows=3)

    st.markdown(f"**Rows:** {profile['row_count']}  **Columns:** {len(profile['columns'])}")
    with st.expander("CSV columns"):
        st.write(profile["columns"])

    model_name = st.text_input("OpenAI model", value=core.DEFAULT_OPENAI_MODEL).strip() or core.DEFAULT_OPENAI_MODEL

    if st.button("Generate OpenAI CSV report package"):
        try:
            report = csv_openai.build_openai_csv_report_package(
                client=core.get_openai_client(),
                csv_path=selected_path,
                output_dir=CSV_PROJECT_DIR,
                model=core.DEFAULT_REPORT_OPENAI_MODEL,
                reasoning_effort=core.DEFAULT_REPORT_REASONING_EFFORT,
            )
            st.success(f"Report package written to {report.parent}")
        except Exception as exc:
            st.error(f"Report failed: {exc}")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"]: List[Dict[str, object]] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(str(msg.get("role", "assistant"))):
            st.write(msg.get("content", ""))

    user_input = st.chat_input("Ask about the selected CSV...")
    if not user_input:
        return

    st.session_state["chat_messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    try:
        reply = csv_openai.answer_csv_question(
            core.get_openai_client(),
            selected_path,
            user_input,
            model=model_name,
        )
    except Exception as exc:
        reply = f"Error while processing your question:\n{exc}"

    st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)


if __name__ == "__main__":
    main()

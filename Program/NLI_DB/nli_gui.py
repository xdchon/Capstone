from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
    from tkinter import scrolledtext
except ModuleNotFoundError:
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    simpledialog = None  # type: ignore[assignment]
    scrolledtext = None  # type: ignore[assignment]

import csv_openai


BASE_DIR = Path(__file__).resolve().parent
CSV_PROJECTS_DIR = BASE_DIR / "csv_projects"
PROJECT_DIR = CSV_PROJECTS_DIR / "default_project"
OPENAI_KEY_FILE = BASE_DIR / "openai_api_key.txt"

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_REPORT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_REPORT_REASONING_EFFORT = "high"


def set_active_csv_project(project_dir: Path) -> None:
    global PROJECT_DIR
    PROJECT_DIR = project_dir
    csv_project_dir()


def csv_project_dir() -> Path:
    data_dir = PROJECT_DIR / "source_csv_files"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def create_empty_csv_project(project_dir: Path) -> None:
    (project_dir / "source_csv_files").mkdir(parents=True, exist_ok=True)


def list_project_csvs() -> List[str]:
    return sorted(path.name for path in csv_project_dir().glob("*.csv") if path.is_file())


def resolve_source_csv_path(source_csv: str) -> Path:
    candidates = [
        csv_project_dir() / source_csv,
        PROJECT_DIR / source_csv,
        BASE_DIR / "CSV_data" / source_csv,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find CSV file '{source_csv}'. Add it to this project again.")


def selected_or_single_csv(selected: str) -> str:
    datasets = list_project_csvs()
    if selected and selected != "All project CSVs":
        return selected
    if len(datasets) == 1:
        return datasets[0]
    if not datasets:
        raise RuntimeError("No CSV files have been added to this project.")
    raise RuntimeError("Choose one CSV dataset first. CSV-direct analysis works on one source CSV at a time.")


def get_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is not installed. Install it with 'pip install openai'.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and OPENAI_KEY_FILE.is_file():
        for line in OPENAI_KEY_FILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                api_key = stripped
                break

    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY or put the key in "
            f"{OPENAI_KEY_FILE}."
        )
    return OpenAI(api_key=api_key)  # type: ignore[call-arg]


def build_report_package(output_dir: Path, source_csv: str | None = None) -> Path:
    selected = source_csv or selected_or_single_csv("All project CSVs")
    csv_path = resolve_source_csv_path(selected)
    return csv_openai.build_openai_csv_report_package(
        client=get_openai_client(),
        csv_path=csv_path,
        output_dir=output_dir,
        model=DEFAULT_REPORT_OPENAI_MODEL,
        reasoning_effort=DEFAULT_REPORT_REASONING_EFFORT,
    )


def _safe_project_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name.strip()).strip("._")
    return safe or "csv_project"


class NLIGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NLI CSV OpenAI Explorer")
        self.root.geometry("980x760")
        set_active_csv_project(PROJECT_DIR)

        project_frame = tk.LabelFrame(root, text="1. CSV project")
        project_frame.pack(fill=tk.X, padx=8, pady=4)

        self.project_var = tk.StringVar(value=f"Active CSV project folder: {PROJECT_DIR}")
        tk.Label(project_frame, textvariable=self.project_var, anchor="w").pack(fill=tk.X, padx=4, pady=2)

        tk.Button(project_frame, text="New Project", command=self.new_project).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(project_frame, text="Open Project", command=self.open_project).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(project_frame, text="Project Summary", command=self.show_project_summary).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(project_frame, text="Generate Report", command=self.generate_report).pack(side=tk.LEFT, padx=4, pady=4)

        model_frame = tk.Frame(root)
        model_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(model_frame, text="OpenAI model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=DEFAULT_OPENAI_MODEL)
        tk.Entry(model_frame, textvariable=self.model_var, width=20).pack(side=tk.LEFT, padx=4)

        import_frame = tk.LabelFrame(root, text="2. Add CSV files")
        import_frame.pack(fill=tk.X, padx=8, pady=4)
        self.csv_label_var = tk.StringVar(value="No CSV selected.")
        tk.Label(import_frame, textvariable=self.csv_label_var, anchor="w").pack(fill=tk.X, padx=4, pady=2)
        tk.Button(import_frame, text="Add CSV Files", command=self.select_and_import_csvs).pack(side=tk.LEFT, padx=4, pady=4)
        self.folder_label_var = tk.StringVar(value=f"CSV folder: {csv_project_dir()}")
        tk.Label(import_frame, textvariable=self.folder_label_var, anchor="w").pack(fill=tk.X, padx=4, pady=2)

        dataset_frame = tk.LabelFrame(root, text="3. Dataset scope")
        dataset_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(dataset_frame, text="CSV dataset:").pack(side=tk.LEFT, padx=4, pady=4)
        self.dataset_filter_var = tk.StringVar(value="All project CSVs")
        self.dataset_menu = tk.OptionMenu(dataset_frame, self.dataset_filter_var, "All project CSVs")
        self.dataset_menu.pack(side=tk.LEFT, padx=4, pady=4)

        qa_frame = tk.LabelFrame(root, text="4. Ask questions")
        qa_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.chat_history = scrolledtext.ScrolledText(qa_frame, height=18, state="disabled")
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 4))

        action_frame = tk.Frame(qa_frame)
        action_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(action_frame, text="Show Columns", command=self.show_schema).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(action_frame, text="Find Correlations", command=self.show_correlations).pack(side=tk.LEFT, padx=(0, 4))

        input_frame = tk.Frame(qa_frame)
        input_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Label(input_frame, text="You:").pack(side=tk.LEFT)
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        self.input_entry.bind("<Return>", self.ask_question)
        tk.Button(input_frame, text="Send", command=self.ask_question).pack(side=tk.LEFT)

        self.refresh_dataset_options()

    def _append_chat(self, text: str) -> None:
        self.chat_history.configure(state="normal")
        self.chat_history.insert(tk.END, text + "\n")
        self.chat_history.see(tk.END)
        self.chat_history.configure(state="disabled")

    def _set_project(self, project_dir: Path) -> None:
        set_active_csv_project(project_dir)
        self.project_var.set(f"Active CSV project folder: {PROJECT_DIR}")
        self.folder_label_var.set(f"CSV folder: {csv_project_dir()}")
        self.csv_label_var.set("No CSV selected.")
        self.refresh_dataset_options()

    def new_project(self) -> None:
        CSV_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        name = simpledialog.askstring("New CSV project", "Project folder name:")
        if not name:
            return
        project_dir = CSV_PROJECTS_DIR / _safe_project_name(name)
        create_empty_csv_project(project_dir)
        self._set_project(project_dir)
        self._append_chat(f"System: Created CSV project {PROJECT_DIR.name}.")

    def open_project(self) -> None:
        CSV_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        folder_path = filedialog.askdirectory(title="Open CSV project folder", initialdir=CSV_PROJECTS_DIR)
        if not folder_path:
            return
        self._set_project(Path(folder_path))
        self._append_chat(f"System: Opened CSV project {PROJECT_DIR.name}.")

    def show_project_summary(self) -> None:
        datasets = list_project_csvs()
        lines = [f"CSV project folder: {PROJECT_DIR}", ""]
        if not datasets:
            lines.append("No CSV files added yet.")
        else:
            lines.append("Project CSV files:")
            for name in datasets:
                try:
                    profile = csv_openai.csv_profile(resolve_source_csv_path(name), preview_rows=0)
                    lines.append(f"  - {name}: {profile['row_count']} rows, {len(profile['columns'])} columns")
                except Exception as exc:
                    lines.append(f"  - {name}: could not read profile ({exc})")
        self._append_chat("System:\n" + "\n".join(lines))

    def show_schema(self) -> None:
        try:
            selected = selected_or_single_csv(self.dataset_filter_var.get().strip())
            profile = csv_openai.csv_profile(resolve_source_csv_path(selected), preview_rows=0)
        except Exception as exc:
            messagebox.showerror("CSV error", f"Could not read selected CSV:\n{exc}")
            return
        columns = "\n".join(f"- {col}" for col in profile["columns"])
        self._append_chat(f"CSV columns for {selected}:\n{columns}")

    def show_correlations(self) -> None:
        try:
            selected = selected_or_single_csv(self.dataset_filter_var.get().strip())
            answer = csv_openai.answer_csv_question(
                get_openai_client(),
                resolve_source_csv_path(selected),
                "Find the strongest relevant non-intensity numeric correlations in this CSV. Focus on size, speed, position, direction, and shape columns where present. Explain caveats clearly.",
                model=self.model_var.get().strip() or DEFAULT_OPENAI_MODEL,
            )
        except Exception as exc:
            messagebox.showerror("Correlation request failed", str(exc))
            return
        self._append_chat(f"Assistant: {answer}")

    def generate_report(self) -> None:
        try:
            selected_dataset = selected_or_single_csv(self.dataset_filter_var.get().strip())
        except Exception as exc:
            messagebox.showerror("Report failed", str(exc))
            return
        output_dir = filedialog.askdirectory(title="Choose folder for analysis report package", initialdir=PROJECT_DIR)
        if not output_dir:
            return
        try:
            report_path = build_report_package(Path(output_dir), source_csv=selected_dataset)
        except Exception as exc:
            messagebox.showerror("Report failed", f"Could not generate report:\n{exc}")
            return
        messagebox.showinfo("Report created", f"Saved report package:\n{report_path.parent}")
        self._append_chat(f"System: Report package saved to {report_path.parent}")

    def refresh_dataset_options(self) -> None:
        options = ["All project CSVs"] + list_project_csvs()
        self.dataset_filter_var.set(options[0])
        menu = self.dataset_menu["menu"]
        menu.delete(0, "end")
        for value in options:
            menu.add_command(label=value, command=lambda v=value: self.dataset_filter_var.set(v))

    def select_and_import_csvs(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="Select one or more CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not file_paths:
            return

        existing_csvs = set(list_project_csvs())
        selected_paths = [Path(file_path) for file_path in file_paths]
        selected_names = [path.name for path in selected_paths]
        duplicate_names = sorted(name for name in selected_names if name in existing_csvs)
        if duplicate_names:
            messagebox.showerror(
                "CSV already added",
                "These CSV filenames are already in this project:\n\n" + "\n".join(duplicate_names),
            )
            return

        imported: List[str] = []
        for path in selected_paths:
            try:
                shutil.copyfile(path, csv_project_dir() / path.name)
            except Exception as exc:
                messagebox.showerror("Import failed", f"Could not copy {path.name}:\n{exc}")
                return
            imported.append(path.name)

        self.csv_label_var.set(f"Added {len(imported)} CSV file(s).")
        self.refresh_dataset_options()
        self._append_chat("System: Added CSV files:\n" + "\n".join(f"- {name}" for name in imported))

    def ask_question(self, event: object | None = None) -> None:
        question = self.input_var.get().strip()
        if not question:
            messagebox.showwarning("No question", "Please type a question first.")
            return
        self._append_chat(f"You: {question}")
        self.input_var.set("")
        self.root.update_idletasks()

        try:
            selected_dataset = selected_or_single_csv(self.dataset_filter_var.get().strip())
            csv_path = resolve_source_csv_path(selected_dataset)
            answer = csv_openai.answer_csv_question(
                get_openai_client(),
                csv_path,
                question,
                model=self.model_var.get().strip() or DEFAULT_OPENAI_MODEL,
            )
        except Exception as exc:
            self._append_chat(f"Assistant: Error while processing question:\n{exc}")
            return

        self._append_chat(f"Assistant: {answer}")
        self._append_chat(f"Analysis details:\n- Source CSV used directly: {csv_path}")


def main() -> None:
    if tk is None:
        raise RuntimeError("tkinter is not installed, so the desktop GUI cannot start.")
    root = tk.Tk()
    NLIGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()

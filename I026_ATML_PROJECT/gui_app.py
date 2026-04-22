from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from checkpoint_registry import (
    DEFAULT_REGISTRY_PATH,
    load_registry,
    resolve_checkpoint_for_profile,
    validate_checkpoint_dir,
)
from generator import build_backend, generate_synthetic, write_csv
from prompt_parser.parse_router import parse_user_prompt
from prompt_parser.parser import PromptParseException

HARD_CODED_BACKEND = "gan_ae"
HARD_CODED_REGISTRY = DEFAULT_REGISTRY_PATH

APP_STYLESHEET = """
QWidget {
    background: #161b22;
    color: #f0f6fc;
    font-family: Helvetica, Arial, sans-serif;
    font-size: 13px;
}
QLabel#titleLabel {
    font-size: 22px;
    font-weight: 700;
}
QTextEdit, QLineEdit, QComboBox {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px;
    color: #f0f6fc;
    selection-background-color: #1f6feb;
}
QGroupBox {
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #8b949e;
}
QPushButton {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 12px;
    color: #f0f6fc;
}
QPushButton:hover {
    background: #30363d;
}
QPushButton:pressed {
    background: #3d444d;
}
QPushButton:disabled {
    color: #6e7681;
    background: #161b22;
    border-color: #30363d;
}
"""


def _timestamped_csv_path(base_path: str | None) -> str:
    raw = base_path.strip() if base_path else "output/synthetic_data.csv"
    path = Path(raw)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    return str(path)


def _open_file_with_system_default(path: str) -> None:
    file_url = QUrl.fromLocalFile(str(Path(path).resolve()))
    if not QDesktopServices.openUrl(file_url):
        raise OSError(f"Could not open file with system default app: {path}")


class PromptGeneratorApp(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Syngen")
        self.resize(860, 620)
        self.setMinimumSize(760, 520)

        self.last_csv_path: str | None = None
        self.parse_mode_combo: QComboBox
        self.out_path_edit: QLineEdit
        self.prompt_text: QTextEdit
        self.output_text: QTextEdit
        self.open_btn: QPushButton

        self._build_ui()

    @staticmethod
    def _row_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setMinimumWidth(150)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return label

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        title = QLabel("Syngen: Prompt-to-CSV Generator")
        title.setObjectName("titleLabel")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title.setFont(title_font)
        root_layout.addWidget(title)

        root_layout.addWidget(QLabel("Enter your prompt:"))
        self.prompt_text = QTextEdit()
        self.prompt_text.setPlaceholderText("Describe what dataset you want to generate...")
        self.prompt_text.setFixedHeight(140)
        self.prompt_text.setPlainText(
            "Generate 1200 healthcare patients age 40-75 with high glucose and hypertension, mostly non smoker"
        )
        root_layout.addWidget(self.prompt_text)

        options_group = QGroupBox("Generation Options")
        options_layout = QVBoxLayout(options_group)
        options_layout.setContentsMargins(14, 18, 14, 12)
        options_layout.setSpacing(10)

        parse_row = QHBoxLayout()
        parse_row.setSpacing(12)
        parse_row.addWidget(self._row_label("Parse mode"))
        self.parse_mode_combo = QComboBox()
        self.parse_mode_combo.addItems(["hybrid", "rules", "llm"])
        self.parse_mode_combo.setCurrentText("hybrid")
        self.parse_mode_combo.setMinimumWidth(220)
        parse_row.addWidget(self.parse_mode_combo, 1)
        options_layout.addLayout(parse_row)

        backend_row = QHBoxLayout()
        backend_row.setSpacing(12)
        backend_row.addWidget(self._row_label("Backend"))
        backend_value = QLabel("gan_ae (hard-coded)")
        backend_value.setWordWrap(True)
        backend_row.addWidget(backend_value, 1)
        options_layout.addLayout(backend_row)

        output_row = QHBoxLayout()
        output_row.setSpacing(12)
        output_row.addWidget(self._row_label("Output base path"))
        self.out_path_edit = QLineEdit("output/synthetic_data.csv")
        output_row.addWidget(self.out_path_edit, 1)
        options_layout.addLayout(output_row)

        root_layout.addWidget(options_group)

        routing_group = QGroupBox("Checkpoint Routing")
        routing_layout = QVBoxLayout(routing_group)
        routing_layout.setContentsMargins(14, 18, 14, 12)
        routing_layout.setSpacing(8)
        registry_row = QHBoxLayout()
        registry_row.setSpacing(12)
        registry_row.addWidget(self._row_label("Registry"))
        registry_value = QLabel(HARD_CODED_REGISTRY)
        registry_value.setWordWrap(True)
        registry_row.addWidget(registry_value, 1)
        routing_layout.addLayout(registry_row)
        root_layout.addWidget(routing_group)

        actions_layout = QHBoxLayout()
        parse_btn = QPushButton("Parse Preview")
        parse_btn.clicked.connect(self.on_preview)
        actions_layout.addWidget(parse_btn)

        generate_btn = QPushButton("Generate Dataset")
        generate_btn.clicked.connect(self.on_generate)
        actions_layout.addWidget(generate_btn)

        self.open_btn = QPushButton("Open Last CSV")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.on_open_last)
        actions_layout.addWidget(self.open_btn)
        actions_layout.addStretch(1)
        root_layout.addLayout(actions_layout)

        root_layout.addWidget(QLabel("Result / Warnings:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(220)
        self.output_text.setPlainText(
            "Ready.\n\n"
            "- Enter your prompt above\n"
            "- Click Parse Preview to inspect parsed constraints\n"
            "- Click Generate Dataset to create a CSV"
        )
        root_layout.addWidget(self.output_text)

    def on_generate(self) -> None:
        prompt = self.prompt_text.toPlainText().strip()
        if not prompt:
            QMessageBox.critical(self, "Missing Prompt", "Please enter a prompt.")
            return

        try:
            spec = parse_user_prompt(prompt, mode=self.parse_mode_combo.currentText())
            registry = load_registry(HARD_CODED_REGISTRY)
            resolved_ckpt = resolve_checkpoint_for_profile(
                spec.target_dataset_profile,
                registry,
                prompt_text=prompt,
            )
            missing = validate_checkpoint_dir(resolved_ckpt)
            if missing:
                QMessageBox.critical(
                    self,
                    "Missing GAN+AE Checkpoints",
                    "Full GAN+AE artifacts not found in selected checkpoint:\n"
                    f"{resolved_ckpt}\n\n"
                    "Missing files:\n- "
                    + "\n- ".join(missing)
                    + "\n\nTrain first with:\n"
                    "python -m gan_ae_full.train --csv path/to/train.csv --out checkpoints/full_gan_ae_demo --device cpu\n\n"
                    "Then map your domain in checkpoints/registry.json",
                )
                return
            backend = build_backend(HARD_CODED_BACKEND, resolved_ckpt)
            out_path = _timestamped_csv_path(self.out_path_edit.text())
            rows = generate_synthetic(spec, backend=backend)
            csv_path = write_csv(rows, out_path)
            self.last_csv_path = csv_path
            self.open_btn.setEnabled(True)
            self._write_result(spec, csv_path, len(rows), resolved_ckpt)
            self._ask_open_excel(csv_path)
        except PromptParseException as exc:
            QMessageBox.critical(self, "Generation Failed", self._format_parse_error(exc))
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Generation Failed", str(exc))
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "Unexpected Error", str(exc))

    def on_preview(self) -> None:
        prompt = self.prompt_text.toPlainText().strip()
        if not prompt:
            QMessageBox.critical(self, "Missing Prompt", "Please enter a prompt.")
            return
        try:
            spec = parse_user_prompt(prompt, mode=self.parse_mode_combo.currentText())
            preview = {
                "n_rows": spec.n_rows,
                "target_dataset_profile": spec.target_dataset_profile,
                "strict_mode": spec.strict_mode,
                "seed": spec.seed,
                "filters": spec.filters,
                "distribution_hints": spec.distribution_hints,
                "priority_rules": spec.priority_rules,
                "warnings": spec.warnings,
            }
            self.output_text.setPlainText("Parsed prompt preview:\n\n" + json.dumps(preview, indent=2))
        except PromptParseException as exc:
            QMessageBox.critical(self, "Parse Failed", self._format_parse_error(exc))
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "Parse Failed", str(exc))

    def on_open_last(self) -> None:
        if not self.last_csv_path:
            QMessageBox.information(self, "No File", "No generated CSV available yet.")
            return
        try:
            _open_file_with_system_default(self.last_csv_path)
        except Exception as exc:
            QMessageBox.warning(self, "Open Failed", f"Could not open CSV automatically:\n{exc}")

    def _write_result(
        self,
        spec,
        csv_path: str,
        n_rows: int,
        ckpt_path: str,
    ) -> None:
        lines = [
            f"CSV created: {csv_path}",
            f"Rows generated: {n_rows}",
            f"Profile: {spec.target_dataset_profile}",
            f"Checkpoint used: {ckpt_path}",
            "",
            "Parsed prompt summary:",
            json.dumps(
                {
                    "n_rows": spec.n_rows,
                    "strict_mode": spec.strict_mode,
                    "seed": spec.seed,
                    "filters": spec.filters,
                    "distribution_hints": spec.distribution_hints,
                },
                indent=2,
            ),
        ]
        if spec.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {w}" for w in spec.warnings)
        else:
            lines.append("Warnings: none")
        self.output_text.setPlainText("\n".join(lines))

    def _format_parse_error(self, exc: PromptParseException) -> str:
        rep = exc.report
        parts = [rep.message]
        if rep.conflicting_fields:
            parts.append("Conflicting fields: " + ", ".join(rep.conflicting_fields))
        if rep.offending_clauses:
            parts.append("Offending clauses: " + " | ".join(rep.offending_clauses))
        if rep.suggested_prompt:
            parts.append("Suggested fix: " + rep.suggested_prompt)
        if rep.warnings:
            parts.append("Warnings: " + " | ".join(rep.warnings))
        return "\n\n".join(parts)

    def _ask_open_excel(self, csv_path: str) -> None:
        open_now = QMessageBox.question(
            self,
            "CSV Ready",
            f"Dataset saved to:\n{csv_path}\n\nOpen with system default app now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if open_now != QMessageBox.Yes:
            return
        try:
            _open_file_with_system_default(csv_path)
        except Exception as exc:
            QMessageBox.warning(self, "Open Failed", f"Could not open CSV automatically:\n{exc}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = PromptGeneratorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

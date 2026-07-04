from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QProcess, QProcessEnvironment, QThread, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QInputDialog,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from panodeon.core.colmap_export import ColmapExportSettings, export_item_for_colmap, write_colmap_metadata
from panodeon.core.image_io import IMAGE_EXTENSIONS, load_mask, load_rgb, save_mask
from panodeon.core.mask_ops import mask_area
from panodeon.core.overlay import overlay_mask
from panodeon.core.project_state import ImageItem, ProjectState
from panodeon.core.video_extract import extract_video_frames, is_video_path
from panodeon.generators.base import MaskOptions
from panodeon.generators.cubemap import CubemapGenerator, merge_compare_masks
from panodeon.generators.direct import DirectEquirectangularGenerator
from panodeon.generators.subprocess_cubemap import PersistentCubemapGenerator
from panodeon.inference.deim_wholebody import DeimWholebodySegmenter
from panodeon.inference.providers import available_onnx_providers, provider_label, resolve_execution_providers, selectable_onnx_providers
from panodeon.tools.run_colmap import (
    ColmapRunSettings,
    ColmapStep,
    build_colmap_steps,
    database_has_rows,
    detect_colmap_gpu_options,
    detect_colmap_mapper_options,
    dense_model_exists,
    feature_extraction_done,
    should_overwrite_outputs,
    sparse_model_exists,
    validate_export_dir,
)
from panodeon.tools.align_colmap_stella_rot import align_colmap_model_to_stella_up
from panodeon.sampler import events as sampler_events
from panodeon.sampler.workflow import (
    PipelineSettings,
    PipelineStep,
    build_pipeline_steps,
    project_root as sampler_project_root,
)
from panodeon.sampler.resume import load_workflow_state, step_is_complete, step_key, step_signature
from panodeon.ui.image_canvas import ImageCanvas
from panodeon.ui.workers import TaskWorker


MODERN_STYLE = """
QMainWindow {
    background-color: #2e2e2e;
}
QDialog, QMessageBox {
    background-color: #2e2e2e;
}
QWidget {
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Inter', 'Meiryo', sans-serif;
    font-size: 9pt;
}
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QSplitter::handle {
    background-color: #1d1d1d;
}
QSplitter::handle:horizontal {
    width: 2px;
}
QGroupBox {
    border: 1px solid #1d1d1d;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: bold;
    color: #e57d22;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    background-color: #2e2e2e;
}
QPushButton {
    background-color: #545454;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
    color: #ffffff;
}
QPushButton:hover {
    background-color: #646464;
    border-color: #e57d22;
}
QPushButton:pressed {
    background-color: #3e3e3e;
}
QPushButton:disabled {
    background-color: #2e2e2e;
    color: #808080;
    border-color: #1d1d1d;
}
QPushButton#primaryButton {
    background-color: #545454;
    border: 1px solid #e57d22;
    color: #ffffff;
}
QPushButton#primaryButton:hover {
    background-color: #e57d22;
    border-color: #f2984b;
    color: #ffffff;
}
QPushButton#primaryButton:pressed {
    background-color: #b85c13;
}
QPushButton#toolButton {
    color: #78a3c8;
}
QComboBox {
    background-color: #1d1d1d;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
}
QComboBox:hover {
    border-color: #e57d22;
}
QComboBox::drop-down {
    border-left: 1px solid #2e2e2e;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #1d1d1d;
    border: 1px solid #2e2e2e;
    selection-background-color: #e57d22;
    selection-color: #ffffff;
}
QSpinBox {
    background-color: #1d1d1d;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
    color: #ffffff;
}
QSpinBox:hover {
    border-color: #e57d22;
}
QLineEdit, QPlainTextEdit {
    background-color: #1d1d1d;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    padding: 4px 8px;
    color: #ffffff;
}
QProgressBar {
    background-color: #1d1d1d;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    text-align: center;
    color: #ffffff;
}
QProgressBar::chunk {
    background-color: #e57d22;
    border-radius: 3px;
}
QListWidget {
    background-color: #1d1d1d;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    padding: 4px;
}
QListWidget::item {
    border-radius: 4px;
    padding: 6px 8px;
    margin: 2px 0px;
}
QListWidget::item:hover {
    background-color: #3e3e3e;
}
QListWidget::item:selected {
    background-color: #e57d22;
    color: #ffffff;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #2e2e2e;
    border-radius: 3px;
    background-color: #1d1d1d;
}
QCheckBox::indicator:hover {
    border-color: #e57d22;
}
QCheckBox::indicator:checked {
    background-color: #e57d22;
    border-color: #f2984b;
}
QTabWidget::pane {
    border: 1px solid #1d1d1d;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #1d1d1d;
    border: 1px solid #2e2e2e;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 6px 12px;
}
QTabBar::tab:selected {
    background-color: #545454;
    color: #ffffff;
}
QTabBar::tab:hover {
    border-color: #e57d22;
}
QScrollBar:vertical {
    background: #1d1d1d;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #545454;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #646464;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    background: none;
    height: 0px;
}
QLabel {
    color: #ffffff;
}
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Panodeon")
        self.setAcceptDrops(True)
        self.setStyleSheet(MODERN_STYLE)
        self.state = ProjectState()
        self.current_item: ImageItem | None = None
        self.current_image: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.mask_dirty = False
        self._restoring_list_row = False
        self.undo_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []
        self.disabled_provider_names: set[str] = set()
        self.direct_generator = DirectEquirectangularGenerator()
        self.cubemap_generator = CubemapGenerator()
        self.segmenter = None
        self.worker_thread: QThread | None = None
        self.worker: TaskWorker | None = None
        self.colmap_process: QProcess | None = None
        self.colmap_steps: list[ColmapStep] = []
        self.colmap_step_index = 0
        self._colmap_cancelled = False
        self.sampler_process: QProcess | None = None
        self.sampler_steps: list[PipelineStep] = []
        self.sampler_step_index = 0
        self._sampler_cancelled = False
        self.sampler_output_dir: Path | None = None
        self._sampler_stdout_buffer = ""
        self._busy = False
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.refresh_preview)
        self.colmap_status_timer = QTimer(self)
        self.colmap_status_timer.setInterval(5000)
        self.colmap_status_timer.timeout.connect(self.refresh_colmap_export_info)
        self._build_ui()
        self.refresh_model_list()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Top-level tabs. New first-level tabs can be added with a single
        # self.tabs.addTab(...) call below; each tab is a plain container widget.
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)

        mask_tab = QWidget()
        mask_tab_layout = QHBoxLayout(mask_tab)
        mask_tab_layout.setContentsMargins(8, 8, 8, 8)
        mask_tab_layout.setSpacing(8)
        colmap_tab = QWidget()
        colmap_layout = QVBoxLayout(colmap_tab)
        colmap_layout.setContentsMargins(8, 8, 8, 8)
        colmap_layout.setSpacing(8)
        # The COLMAP tab is a tall stack of forms; wrap it in a scroll area so
        # it does not impose a large minimum height on the whole window (the
        # tab widget's minimum height is that of its tallest tab).
        colmap_scroll = QScrollArea()
        colmap_scroll.setWidget(colmap_tab)
        colmap_scroll.setWidgetResizable(True)
        colmap_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        colmap_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Sampler tab: like the COLMAP tab it is a tall form stack, so wrap it in a
        # scroll area to keep the window from inheriting a large minimum height.
        # It is the leftmost tab because it is the first step of the workflow
        # (sample frames from a 360 video before masking and COLMAP export).
        sampler_tab = self._build_sampler_tab()
        sampler_scroll = QScrollArea()
        sampler_scroll.setWidget(sampler_tab)
        sampler_scroll.setWidgetResizable(True)
        sampler_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        sampler_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tabs.addTab(sampler_scroll, "Sampler")
        self.tabs.addTab(mask_tab, "Mask")
        self.tabs.addTab(colmap_scroll, "COLMAP")

        # Mask tab: three resizable columns (image list / canvas / settings).
        mask_splitter = QSplitter(Qt.Orientation.Horizontal)
        mask_tab_layout.addWidget(mask_splitter)

        left = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        self.open_button = QPushButton("Open Folder")
        self.open_button.setObjectName("primaryButton")
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        left_layout.addWidget(self.open_button)
        left_layout.addWidget(self.image_list)
        left.setLayout(left_layout)
        mask_splitter.addWidget(left)

        self.canvas = ImageCanvas()
        self.canvas.setStyleSheet("border: 1px solid #1d1d1d; border-radius: 4px;")
        mask_splitter.addWidget(self.canvas)

        right = QWidget()
        mask_layout = QVBoxLayout(right)
        mask_layout.setContentsMargins(0, 0, 0, 0)
        mask_layout.setSpacing(8)

        # 1. AI Model Config Group
        group_model = QGroupBox("AI Model Settings")
        layout_model = QFormLayout(group_model)
        layout_model.setContentsMargins(10, 15, 10, 10)
        layout_model.setSpacing(8)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["direct", "cubemap", "both"])
        self.strategy_combo.setCurrentText("both")
        self.model_combo = QComboBox()
        self.provider_combo = QComboBox()
        self.refresh_provider_list()

        self.model_button = QPushButton("Refresh")
        model_layout = QHBoxLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(4)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.model_button)

        layout_model.addRow("Mode", self.strategy_combo)
        layout_model.addRow("ONNX Model", model_layout)
        layout_model.addRow("EP Provider", self.provider_combo)
        mask_layout.addWidget(group_model)

        # 2. AI Parameters Group
        group_params = QGroupBox("Detection Parameters")
        layout_params = QFormLayout(group_params)
        layout_params.setContentsMargins(10, 15, 10, 10)
        layout_params.setSpacing(8)

        self.score_spin = QSpinBox()
        self.score_spin.setRange(1, 99)
        self.score_spin.setValue(35)
        self.mask_threshold_spin = QSpinBox()
        self.mask_threshold_spin.setRange(1, 99)
        self.mask_threshold_spin.setValue(40)

        layout_params.addRow("Conf Score %", self.score_spin)
        layout_params.addRow("Mask Thresh %", self.mask_threshold_spin)
        mask_layout.addWidget(group_params)

        # 3. Manual Mask Editor Group
        group_manual = QGroupBox("Manual Mask Editor")
        layout_manual = QFormLayout(group_manual)
        layout_manual.setContentsMargins(10, 15, 10, 10)
        layout_manual.setSpacing(8)

        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 256)
        self.brush_spin.setValue(24)
        self.erase_check = QCheckBox("Eraser Mode")
        self.overlay_opacity_spin = QSpinBox()
        self.overlay_opacity_spin.setRange(0, 100)
        self.overlay_opacity_spin.setValue(35)

        layout_manual.addRow("Brush Size", self.brush_spin)
        layout_manual.addRow("", self.erase_check)
        layout_manual.addRow("Red Overlay", self.overlay_opacity_spin)
        mask_layout.addWidget(group_manual)

        # 4. Export Settings Group
        group_export = QGroupBox("Export Settings")
        layout_export = QFormLayout(group_export)
        layout_export.setContentsMargins(10, 15, 10, 10)
        layout_export.setSpacing(8)

        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(256, 8192)
        self.tile_size_spin.setSingleStep(256)
        self.tile_size_spin.setValue(3072)
        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(30, 150)
        self.fov_spin.setValue(90)

        layout_export.addRow("Tile Size", self.tile_size_spin)
        layout_export.addRow("FOV deg", self.fov_spin)
        mask_layout.addWidget(group_export)

        group_colmap = QGroupBox("COLMAP")
        layout_colmap = QFormLayout(group_colmap)
        layout_colmap.setContentsMargins(10, 15, 10, 10)
        layout_colmap.setSpacing(8)

        self.colmap_path_edit = QLineEdit(default_colmap_executable())
        self.colmap_browse_button = QPushButton("...")
        self.colmap_browse_button.setObjectName("toolButton")
        colmap_path_layout = QHBoxLayout()
        colmap_path_layout.setContentsMargins(0, 0, 0, 0)
        colmap_path_layout.setSpacing(4)
        colmap_path_layout.addWidget(self.colmap_path_edit)
        colmap_path_layout.addWidget(self.colmap_browse_button)

        self.colmap_matcher_combo = QComboBox()
        self.colmap_matcher_combo.addItems(["sequential", "pairs", "exhaustive", "vocab_tree"])
        self.colmap_sparse_mapper_combo = QComboBox()
        self.colmap_sparse_mapper_combo.addItems(["mapper", "hierarchical_mapper"])
        self.colmap_overwrite_check = QCheckBox("Overwrite outputs")
        self.colmap_overwrite_check.setChecked(True)
        self.colmap_skip_completed_check = QCheckBox("Skip completed steps")
        self.colmap_skip_mapping_check = QCheckBox("Skip mapping")
        self.colmap_rig_ba_check = QCheckBox("Rig bundle adjustment")
        self.colmap_dense_check = QCheckBox("Dense reconstruction")
        self.colmap_use_gpu_check = QCheckBox("Use GPU if available")
        self.colmap_use_gpu_check.setChecked(True)
        self.colmap_gpu_index_edit = QLineEdit("-1")
        self.colmap_snapshot_check = QCheckBox("Mapper snapshots")
        self.colmap_snapshot_check.setChecked(True)
        self.colmap_snapshot_freq_spin = QSpinBox()
        self.colmap_snapshot_freq_spin.setRange(1, 10000)
        self.colmap_snapshot_freq_spin.setValue(50)
        self.colmap_image_path_label = QLabel("-")
        self.colmap_image_path_label.setWordWrap(True)
        self.colmap_image_count_label = QLabel("0")
        self.colmap_mask_count_label = QLabel("0")
        self.colmap_registered_count_label = QLabel("0")
        self.colmap_resume_label = QLabel("-")
        self.colmap_resume_label.setWordWrap(True)
        self.colmap_run_start_label = QLabel("-")
        self.colmap_run_start_label.setWordWrap(True)
        self.colmap_progress = QProgressBar()
        self.colmap_progress.setRange(0, 1)
        self.colmap_progress.setValue(0)
        self.colmap_stage_label = QLabel("Idle")
        self.colmap_stage_label.setWordWrap(True)
        self.colmap_log = QPlainTextEdit()
        self.colmap_log.setReadOnly(True)
        self.colmap_log.setMaximumHeight(120)
        self.run_colmap_button = QPushButton("Run COLMAP")
        self.run_colmap_button.setObjectName("primaryButton")
        self.align_stella_button = QPushButton("Align Stella Up")

        layout_colmap.addRow("Executable", colmap_path_layout)
        layout_colmap.addRow("Matcher", self.colmap_matcher_combo)
        layout_colmap.addRow("Sparse Mapper", self.colmap_sparse_mapper_combo)
        layout_colmap.addRow("", self.colmap_overwrite_check)
        layout_colmap.addRow("", self.colmap_skip_completed_check)
        layout_colmap.addRow("", self.colmap_skip_mapping_check)
        layout_colmap.addRow("", self.colmap_rig_ba_check)
        layout_colmap.addRow("", self.colmap_dense_check)
        layout_colmap.addRow("", self.colmap_use_gpu_check)
        layout_colmap.addRow("GPU Index", self.colmap_gpu_index_edit)
        layout_colmap.addRow("", self.colmap_snapshot_check)
        layout_colmap.addRow("Snapshot Freq", self.colmap_snapshot_freq_spin)
        layout_colmap.addRow("Image Path", self.colmap_image_path_label)
        layout_colmap.addRow("Images", self.colmap_image_count_label)
        layout_colmap.addRow("Masks", self.colmap_mask_count_label)
        layout_colmap.addRow("Registered", self.colmap_registered_count_label)
        layout_colmap.addRow("Progress", self.colmap_progress)
        layout_colmap.addRow("Stage", self.colmap_stage_label)
        layout_colmap.addRow("Log", self.colmap_log)
        layout_colmap.addRow("", self.run_colmap_button)
        layout_colmap.addRow("", self.align_stella_button)
        colmap_layout.addWidget(group_colmap)

        group_colmap_status = QGroupBox("COLMAP Status")
        layout_colmap_status = QFormLayout(group_colmap_status)
        layout_colmap_status.setContentsMargins(10, 15, 10, 10)
        layout_colmap_status.setSpacing(8)
        self.colmap_export_status_label = QLabel("-")
        self.colmap_feature_status_label = QLabel("-")
        self.colmap_rig_status_label = QLabel("-")
        self.colmap_match_status_label = QLabel("-")
        self.colmap_sparse_status_label = QLabel("-")
        self.colmap_rig_ba_status_label = QLabel("-")
        self.colmap_dense_status_label = QLabel("-")
        self.colmap_snapshot_status_label = QLabel("-")
        self.colmap_stella_rot_status_label = QLabel("-")
        for label in (
            self.colmap_export_status_label,
            self.colmap_feature_status_label,
            self.colmap_rig_status_label,
            self.colmap_match_status_label,
            self.colmap_sparse_status_label,
            self.colmap_rig_ba_status_label,
            self.colmap_dense_status_label,
            self.colmap_snapshot_status_label,
            self.colmap_stella_rot_status_label,
        ):
            label.setWordWrap(True)
        layout_colmap_status.addRow("Resume From", self.colmap_resume_label)
        layout_colmap_status.addRow("Run Starts", self.colmap_run_start_label)
        layout_colmap_status.addRow("Export", self.colmap_export_status_label)
        layout_colmap_status.addRow("Features", self.colmap_feature_status_label)
        layout_colmap_status.addRow("Rig", self.colmap_rig_status_label)
        layout_colmap_status.addRow("Matches", self.colmap_match_status_label)
        layout_colmap_status.addRow("Sparse", self.colmap_sparse_status_label)
        layout_colmap_status.addRow("Rig BA", self.colmap_rig_ba_status_label)
        layout_colmap_status.addRow("Dense", self.colmap_dense_status_label)
        layout_colmap_status.addRow("Snapshots", self.colmap_snapshot_status_label)
        layout_colmap_status.addRow("Stella Rot", self.colmap_stella_rot_status_label)
        colmap_layout.addWidget(group_colmap_status)
        colmap_layout.addStretch()

        # Actions area (horizontal layouts)
        ai_buttons = QHBoxLayout()
        ai_buttons.setSpacing(6)
        self.generate_selected_button = QPushButton("Generate Select")
        self.generate_selected_button.setObjectName("primaryButton")
        self.generate_all_button = QPushButton("Generate All")
        ai_buttons.addWidget(self.generate_selected_button)
        ai_buttons.addWidget(self.generate_all_button)
        mask_layout.addLayout(ai_buttons)

        edit_buttons = QHBoxLayout()
        edit_buttons.setSpacing(6)
        self.undo_button = QPushButton("Undo")
        self.undo_button.setObjectName("toolButton")
        self.redo_button = QPushButton("Redo")
        self.redo_button.setObjectName("toolButton")
        self.save_mask_button = QPushButton("Save Mask")
        self.save_mask_button.setObjectName("primaryButton")
        edit_buttons.addWidget(self.undo_button)
        edit_buttons.addWidget(self.redo_button)
        edit_buttons.addWidget(self.save_mask_button)
        mask_layout.addLayout(edit_buttons)

        export_buttons = QHBoxLayout()
        export_buttons.setSpacing(6)
        self.export_selected_button = QPushButton("Export Select")
        self.export_selected_button.setObjectName("primaryButton")
        self.export_all_button = QPushButton("Export All")
        export_buttons.addWidget(self.export_selected_button)
        export_buttons.addWidget(self.export_all_button)
        mask_layout.addLayout(export_buttons)
        mask_layout.addStretch()

        # Wrap the settings column in a scroll area so the window can shrink
        # vertically below the column's natural minimum height; the controls
        # scroll instead of forcing the whole window taller.
        right_scroll = QScrollArea()
        right_scroll.setWidget(right)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mask_splitter.addWidget(right_scroll)
        mask_splitter.setSizes([220, 820, 280])

        # Status label and cancel button share a single bottom row across all
        # tabs: status fills the left, the cancel button stays fixed on the right.
        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(0, 0, 0, 0)
        status_bar.setSpacing(8)

        self.status_label = QLabel("No folder")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #808080; font-size: 8pt; padding: 4px;")
        status_bar.addWidget(self.status_label, 1)

        self.cancel_button = QPushButton("Cancel Task")
        self.cancel_button.setObjectName("toolButton")
        self.cancel_button.setEnabled(False)
        status_bar.addWidget(self.cancel_button, 0)

        main_layout.addLayout(status_bar)

        self.open_button.clicked.connect(self.open_folder)
        self.model_button.clicked.connect(self.refresh_model_list)
        self.model_combo.currentIndexChanged.connect(lambda index: self.load_segmenter())
        self.provider_combo.currentIndexChanged.connect(lambda index: self.load_segmenter())
        self.image_list.currentItemChanged.connect(self.select_image)
        self.canvas.strokeStarted.connect(self.begin_mask_edit)
        self.canvas.strokeCommitted.connect(self.commit_canvas_stroke)
        self.brush_spin.valueChanged.connect(self.update_brush)
        self.erase_check.stateChanged.connect(self.update_brush)
        self.overlay_opacity_spin.valueChanged.connect(lambda value: self.schedule_preview_refresh())
        self.generate_selected_button.clicked.connect(self.generate_selected)
        self.generate_all_button.clicked.connect(self.generate_all)
        self.save_mask_button.clicked.connect(self.save_current_mask)
        self.export_selected_button.clicked.connect(self.export_selected)
        self.export_all_button.clicked.connect(self.export_all)
        self.colmap_browse_button.clicked.connect(self.browse_colmap_executable)
        self.run_colmap_button.clicked.connect(self.run_colmap_gui)
        self.align_stella_button.clicked.connect(self.align_colmap_to_stella_up)
        self.colmap_overwrite_check.stateChanged.connect(lambda value: self.refresh_colmap_export_info())
        self.colmap_skip_completed_check.stateChanged.connect(lambda value: self.refresh_colmap_export_info())
        self.colmap_rig_ba_check.stateChanged.connect(lambda value: self.refresh_colmap_export_info())
        self.colmap_dense_check.stateChanged.connect(lambda value: self.refresh_colmap_export_info())
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.cancel_button.clicked.connect(self.cancel_current_task)
        self.sampler_video_browse_button.clicked.connect(self.browse_sampler_video)
        self.sampler_output_browse_button.clicked.connect(self.browse_sampler_output)
        self.sampler_exe_browse_button.clicked.connect(self.browse_sampler_executable)
        self.sampler_vocab_browse_button.clicked.connect(self.browse_sampler_vocabulary)
        self.sampler_camera_browse_button.clicked.connect(self.browse_sampler_camera_config)
        self.sampler_config_browse_button.clicked.connect(self.browse_sampler_config)
        self.run_sampler_button.clicked.connect(self.run_sampler_gui)
        self.sampler_visualize_button.clicked.connect(self.open_sampler_visualization)
        self.update_brush()
        self._set_tooltips()
        self._update_generate_buttons()

    def _build_sampler_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        group_input = QGroupBox("Frame Sampler")
        form = QFormLayout(group_input)
        form.setContentsMargins(10, 15, 10, 10)
        form.setSpacing(8)

        self.sampler_video_edit = QLineEdit()
        self.sampler_video_browse_button = QPushButton("...")
        self.sampler_video_browse_button.setObjectName("toolButton")
        video_row = QHBoxLayout()
        video_row.setContentsMargins(0, 0, 0, 0)
        video_row.setSpacing(4)
        video_row.addWidget(self.sampler_video_edit)
        video_row.addWidget(self.sampler_video_browse_button)

        self.sampler_output_edit = QLineEdit()
        self.sampler_output_browse_button = QPushButton("...")
        self.sampler_output_browse_button.setObjectName("toolButton")
        output_row = QHBoxLayout()
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.setSpacing(4)
        output_row.addWidget(self.sampler_output_edit)
        output_row.addWidget(self.sampler_output_browse_button)

        self.sampler_frame_skip_spin = QSpinBox()
        self.sampler_frame_skip_spin.setRange(1, 30)
        self.sampler_frame_skip_spin.setValue(2)
        self.sampler_format_combo = QComboBox()
        self.sampler_format_combo.addItems(["jpg", "png"])
        self.sampler_jpeg_quality_spin = QSpinBox()
        self.sampler_jpeg_quality_spin.setRange(1, 100)
        self.sampler_jpeg_quality_spin.setValue(95)

        form.addRow("Video", video_row)
        form.addRow("Output", output_row)
        form.addRow("Frame Skip", self.sampler_frame_skip_spin)
        form.addRow("Image Format", self.sampler_format_combo)
        form.addRow("JPEG Quality", self.sampler_jpeg_quality_spin)
        layout.addWidget(group_input)

        group_advanced = QGroupBox("Sampler Advanced")
        adv = QFormLayout(group_advanced)
        adv.setContentsMargins(10, 15, 10, 10)
        adv.setSpacing(8)

        self.sampler_exe_edit = QLineEdit(default_sampler_executable())
        self.sampler_exe_browse_button = QPushButton("...")
        self.sampler_exe_browse_button.setObjectName("toolButton")
        exe_row = QHBoxLayout()
        exe_row.setContentsMargins(0, 0, 0, 0)
        exe_row.setSpacing(4)
        exe_row.addWidget(self.sampler_exe_edit)
        exe_row.addWidget(self.sampler_exe_browse_button)

        self.sampler_vocab_edit = QLineEdit(default_sampler_vocabulary())
        self.sampler_vocab_browse_button = QPushButton("...")
        self.sampler_vocab_browse_button.setObjectName("toolButton")
        vocab_row = QHBoxLayout()
        vocab_row.setContentsMargins(0, 0, 0, 0)
        vocab_row.setSpacing(4)
        vocab_row.addWidget(self.sampler_vocab_edit)
        vocab_row.addWidget(self.sampler_vocab_browse_button)

        self.sampler_camera_edit = QLineEdit()
        self.sampler_camera_browse_button = QPushButton("...")
        self.sampler_camera_browse_button.setObjectName("toolButton")
        camera_row = QHBoxLayout()
        camera_row.setContentsMargins(0, 0, 0, 0)
        camera_row.setSpacing(4)
        camera_row.addWidget(self.sampler_camera_edit)
        camera_row.addWidget(self.sampler_camera_browse_button)

        self.sampler_config_edit = QLineEdit()
        self.sampler_config_browse_button = QPushButton("...")
        self.sampler_config_browse_button.setObjectName("toolButton")
        config_row = QHBoxLayout()
        config_row.setContentsMargins(0, 0, 0, 0)
        config_row.setSpacing(4)
        config_row.addWidget(self.sampler_config_edit)
        config_row.addWidget(self.sampler_config_browse_button)

        adv.addRow("SLAM Exe", exe_row)
        adv.addRow("ORB Vocab", vocab_row)
        adv.addRow("Camera Config", camera_row)
        adv.addRow("Sampler Config", config_row)
        layout.addWidget(group_advanced)

        group_run = QGroupBox("Sampler Run")
        run_form = QFormLayout(group_run)
        run_form.setContentsMargins(10, 15, 10, 10)
        run_form.setSpacing(8)
        self.sampler_progress = QProgressBar()
        self.sampler_progress.setRange(0, 1000)
        self.sampler_progress.setValue(0)
        self.sampler_stage_label = QLabel("Idle")
        self.sampler_stage_label.setWordWrap(True)
        self.sampler_substage_label = QLabel("-")
        self.sampler_substage_label.setWordWrap(True)
        self.sampler_log = QPlainTextEdit()
        self.sampler_log.setReadOnly(True)
        self.sampler_log.setMaximumHeight(160)
        self.run_sampler_button = QPushButton("Run Sampler")
        self.run_sampler_button.setObjectName("primaryButton")
        self.sampler_visualize_button = QPushButton("Show Trajectory")
        self.sampler_visualize_button.setObjectName("toolButton")
        self.sampler_visualize_button.setEnabled(False)
        run_buttons = QHBoxLayout()
        run_buttons.setContentsMargins(0, 0, 0, 0)
        run_buttons.setSpacing(6)
        run_buttons.addWidget(self.run_sampler_button)
        run_buttons.addWidget(self.sampler_visualize_button)

        run_form.addRow("Progress", self.sampler_progress)
        run_form.addRow("Stage", self.sampler_stage_label)
        run_form.addRow("Detail", self.sampler_substage_label)
        run_form.addRow("Log", self.sampler_log)
        run_form.addRow("", run_buttons)
        layout.addWidget(group_run)
        layout.addStretch()
        return tab

    def browse_sampler_video(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self, "Select 360 video", filter="Video (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.insv);;All (*.*)"
        )
        if not path_text:
            return
        self.sampler_video_edit.setText(path_text)
        if not self.sampler_output_edit.text().strip():
            source = Path(path_text)
            self.sampler_output_edit.setText(str(source.parent / f"{source.stem}_frames"))

    def browse_sampler_output(self) -> None:
        path_text = QFileDialog.getExistingDirectory(self, "Select sampler output folder")
        if path_text:
            self.sampler_output_edit.setText(path_text)

    def browse_sampler_executable(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(self, "Select run_video_slam executable")
        if path_text:
            self.sampler_exe_edit.setText(path_text)

    def browse_sampler_vocabulary(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(self, "Select ORB vocabulary", filter="FBoW (*.fbow);;All (*.*)")
        if path_text:
            self.sampler_vocab_edit.setText(path_text)

    def browse_sampler_camera_config(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(self, "Select camera config", filter="YAML (*.yaml *.yml);;All (*.*)")
        if path_text:
            self.sampler_camera_edit.setText(path_text)

    def browse_sampler_config(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(self, "Select sampler config", filter="JSON (*.json);;All (*.*)")
        if path_text:
            self.sampler_config_edit.setText(path_text)

    def _sampler_settings(self) -> PipelineSettings:
        video = self.sampler_video_edit.text().strip()
        output = self.sampler_output_edit.text().strip()
        if not video:
            raise ValueError("Select a video")
        if not output:
            raise ValueError("Select an output folder")
        camera = self.sampler_camera_edit.text().strip()
        config = self.sampler_config_edit.text().strip()
        return PipelineSettings(
            video_path=Path(video),
            output_dir=Path(output),
            executable=Path(self.sampler_exe_edit.text().strip()),
            vocabulary=Path(self.sampler_vocab_edit.text().strip()),
            frame_skip=self.sampler_frame_skip_spin.value(),
            image_format=self.sampler_format_combo.currentText(),
            jpeg_quality=self.sampler_jpeg_quality_spin.value(),
            camera_config=Path(camera) if camera else None,
            sampler_config=Path(config) if config else None,
        )

    def run_sampler_gui(self) -> None:
        if self.sampler_process is not None or self.worker_thread is not None or self.colmap_process is not None:
            self.status_label.setText("Task already running")
            return
        try:
            settings = self._sampler_settings()
            settings.validate()
        except (Exception, SystemExit) as exc:
            self.status_label.setText(short_error(exc))
            QMessageBox.warning(self, "Sampler", str(exc))
            return
        all_steps = build_pipeline_steps(settings, python_executable=Path(sys.executable))
        self.sampler_steps = self._sampler_pending_steps(settings, all_steps)
        self.sampler_output_dir = settings.output_dir
        self.sampler_step_index = 0
        self._sampler_cancelled = False
        self._sampler_stdout_buffer = ""
        self.sampler_log.clear()
        self.sampler_substage_label.setText("-")
        self.sampler_progress.setRange(0, 1000)
        self.sampler_progress.setValue(0)
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.sampler_steps:
            self.sampler_stage_label.setText("Done")
            self.status_label.setText("Sampler already complete")
            self.append_sampler_log("All sampler steps already have output. Nothing to run.")
            self._finish_sampler_run()
            return
        self.set_busy(True)
        self.start_next_sampler_step()

    def _sampler_pending_steps(self, settings: PipelineSettings, steps: tuple[PipelineStep, ...]) -> list[PipelineStep]:
        # Reuse the sampler's own resume signatures so any stage whose outputs already
        # exist (e.g. a user-supplied stella/trajectory.csv) is skipped. This is what
        # lets the pipeline run without run_video_slam.exe when upstream artifacts exist.
        state = load_workflow_state(settings.output_dir)
        pending: list[PipelineStep] = []
        for step in steps:
            try:
                key = step_key(step.name)
                signature = step_signature(settings, key)
                complete = step_is_complete(settings, key, state, signature)
            except (KeyError, OSError, ValueError):
                complete = False
            if complete:
                self.append_sampler_log(f"[再利用] {step.name}")
            else:
                pending.append(step)
        return pending

    def start_next_sampler_step(self) -> None:
        if self.sampler_step_index >= len(self.sampler_steps):
            self.sampler_process = None
            self.set_busy(False)
            self.sampler_stage_label.setText("Done")
            self.status_label.setText("Sampler finished")
            self.append_sampler_log("Sampler finished")
            self._finish_sampler_run()
            return
        step = self.sampler_steps[self.sampler_step_index]
        self.sampler_stage_label.setText(f"{self.sampler_step_index + 1}/{len(self.sampler_steps)} {step.name}")
        self.status_label.setText(step.name)
        self.append_sampler_log(f"> {' '.join(step.command)}")
        self.sampler_progress.setValue(int(step.progress_start * 1000))
        self._sampler_stdout_buffer = ""
        process = QProcess(self)
        process.setWorkingDirectory(str(sampler_project_root()))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("FRAME_SAMPLER_ROOT", str(sampler_project_root()))
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self.read_sampler_stdout)
        process.readyReadStandardError.connect(self.read_sampler_stderr)
        process.finished.connect(self.on_sampler_finished)
        process.errorOccurred.connect(self.on_sampler_error)
        self.sampler_process = process
        process.start(step.command[0], step.command[1:])

    def read_sampler_stdout(self) -> None:
        if self.sampler_process is None:
            return
        self._sampler_stdout_buffer += bytes(self.sampler_process.readAllStandardOutput()).decode(errors="replace")
        # Only parse complete, newline-terminated lines: a partial JSON chunk must not
        # be handed to json.loads. Keep any trailing fragment for the next read.
        while "\n" in self._sampler_stdout_buffer:
            line, self._sampler_stdout_buffer = self._sampler_stdout_buffer.split("\n", 1)
            self._handle_sampler_line(line.rstrip("\r"))

    def _handle_sampler_line(self, line: str) -> None:
        if not line.strip():
            return
        formatted = sampler_events.format_log_line(line)
        if formatted is not None:
            self.append_sampler_log(formatted)
        substage = sampler_events.substage_name(line)
        if substage is not None:
            self.sampler_substage_label.setText(substage)
        extract_status = sampler_events.extract_progress_status(line)
        if extract_status is not None:
            self.sampler_substage_label.setText(extract_status)
        local = sampler_events.local_progress(line)
        if local is not None and self.sampler_step_index < len(self.sampler_steps):
            step = self.sampler_steps[self.sampler_step_index]
            overall = sampler_events.overall_progress(step.progress_start, step.progress_end, local)
            self.sampler_progress.setValue(int(overall * 1000))

    def read_sampler_stderr(self) -> None:
        if self.sampler_process is None:
            return
        self.append_sampler_log(bytes(self.sampler_process.readAllStandardError()).decode(errors="replace").rstrip())

    def append_sampler_log(self, text: str) -> None:
        if text:
            self.sampler_log.appendPlainText(text)

    def on_sampler_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self.sampler_process is None:
            return
        if self._sampler_cancelled:
            self.sampler_process = None
            self.set_busy(False)
            self.sampler_stage_label.setText("Cancelled")
            self.status_label.setText("Sampler cancelled")
            return
        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            self.sampler_process = None
            self.set_busy(False)
            self.sampler_stage_label.setText("Failed")
            self.status_label.setText(f"Sampler failed: exit {exit_code}")
            return
        step = self.sampler_steps[self.sampler_step_index]
        self.sampler_progress.setValue(int(step.progress_end * 1000))
        self.sampler_step_index += 1
        self.sampler_process = None
        self.start_next_sampler_step()

    def on_sampler_error(self, error: QProcess.ProcessError) -> None:
        self.append_sampler_log(f"Sampler process error: {error.name}")
        if error == QProcess.ProcessError.FailedToStart:
            self.sampler_process = None
            self.set_busy(False)
            self.sampler_stage_label.setText("Failed")
            self.status_label.setText("Sampler failed to start. Check the Python environment.")

    def _finish_sampler_run(self) -> None:
        self._refresh_sampler_visualize_button()
        if self.sampler_output_dir is None:
            return
        frames_dir = self.sampler_output_dir / "frames"
        if not frames_dir.is_dir() or not any(frames_dir.iterdir()):
            self.status_label.setText("Sampler produced no frames")
            return
        self.load_folder(frames_dir)

    def _refresh_sampler_visualize_button(self) -> None:
        path = self._sampler_visualization_path()
        self.sampler_visualize_button.setEnabled(path is not None and path.is_file())

    def _sampler_visualization_path(self) -> Path | None:
        if self.sampler_output_dir is None:
            output = self.sampler_output_edit.text().strip()
            if not output:
                return None
            base = Path(output)
        else:
            base = self.sampler_output_dir
        return base / "sampled" / "trajectory_visualization.html"

    def open_sampler_visualization(self) -> None:
        path = self._sampler_visualization_path()
        if path is None or not path.is_file():
            QMessageBox.information(self, "Trajectory", "The trajectory visualization has not been generated yet.")
            self._refresh_sampler_visualize_button()
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _set_tooltips(self) -> None:
        self.open_button.setToolTip("Open a folder containing panorama images.")
        self.strategy_combo.setToolTip("Mask generation method: direct equirectangular, cubemap, or both.")
        self.model_combo.setToolTip("ONNX model loaded from the local models folder.")
        self.provider_combo.setToolTip("ONNX Runtime execution provider. DirectML supports NVIDIA and AMD on Windows.")
        self.model_button.setToolTip("Refresh the ONNX model list from the models folder.")
        self.save_mask_button.setToolTip("Save the current edited mask PNG.")
        self.export_selected_button.setToolTip("Export selected images as COLMAP perspective tiles and masks.")
        self.export_all_button.setToolTip("Export all images as COLMAP perspective tiles and masks.")
        self.cancel_button.setToolTip("Stop the running generate/export task. Finished images are kept.")
        self.undo_button.setToolTip("Undo the last mask edit.")
        self.redo_button.setToolTip("Redo the last undone mask edit.")
        self.erase_check.setToolTip("Erase mask areas instead of adding them.")
        self.overlay_opacity_spin.setToolTip("Opacity of the mask overlay in percent.")
        self.brush_spin.setToolTip("Brush radius in source image pixels.")
        self.score_spin.setToolTip("Minimum detection confidence in percent.")
        self.mask_threshold_spin.setToolTip("Minimum segmentation mask probability in percent.")
        self.tile_size_spin.setToolTip("Square perspective tile size in pixels.")
        self.fov_spin.setToolTip("Perspective tile field of view in degrees.")
        self.colmap_path_edit.setToolTip("COLMAP executable name or full path.")
        self.colmap_browse_button.setToolTip("Select colmap.exe.")
        self.colmap_matcher_combo.setToolTip("COLMAP matcher command.")
        self.colmap_sparse_mapper_combo.setToolTip("Sparse reconstruction command. hierarchical_mapper is experimental for large datasets.")
        self.colmap_overwrite_check.setToolTip("Delete existing database.db, sparse, and selected dense outputs before running COLMAP.")
        self.colmap_skip_completed_check.setToolTip("Skip steps with existing COLMAP outputs. This takes priority over overwrite.")
        self.colmap_skip_mapping_check.setToolTip("Stop after feature extraction, rig configuration, and matching.")
        self.colmap_rig_ba_check.setToolTip("Run bundle_adjuster after sparse mapping and keep output under exports/sparse_rig_ba.")
        self.colmap_dense_check.setToolTip("Run image undistortion, patch-match stereo, and stereo fusion after sparse mapping.")
        self.colmap_use_gpu_check.setToolTip("Use COLMAP GPU SIFT extraction and matching when the binary supports it.")
        self.colmap_gpu_index_edit.setToolTip("COLMAP GPU index. -1 lets COLMAP choose/use all available GPUs.")
        self.colmap_snapshot_check.setToolTip("Save mapper snapshot models under exports/snapshots during sparse mapping.")
        self.colmap_snapshot_freq_spin.setToolTip("Save a mapper snapshot every N newly registered images.")
        self.run_colmap_button.setToolTip("Run COLMAP on the current exports folder.")
        self.align_stella_button.setToolTip("Create exports/sparse_stella_rot/0 by rotating the sparse model to stella up.")
        self.image_list.setToolTip("Images found in the selected folder.")
        self.canvas.setToolTip("Preview canvas. Wheel to zoom. Draw with left mouse button.")
        self.sampler_video_edit.setToolTip("360 video to sample frames from.")
        self.sampler_output_edit.setToolTip("Sampler work folder. Frames load from its frames/ subfolder when done.")
        self.sampler_frame_skip_spin.setToolTip("SLAM frame skip. 2 favours speed, 1 favours accuracy.")
        self.sampler_format_combo.setToolTip("Extracted image format.")
        self.sampler_jpeg_quality_spin.setToolTip("JPEG quality for extracted frames.")
        self.sampler_exe_edit.setToolTip("stella_vslam run_video_slam executable.")
        self.sampler_vocab_edit.setToolTip("ORB vocabulary (.fbow) used by stella_vslam.")
        self.sampler_camera_edit.setToolTip("Optional stella camera config (.yaml). Auto-generated if empty.")
        self.sampler_config_edit.setToolTip("Optional frame-selection config (.json). Defaults are used if empty.")
        self.run_sampler_button.setToolTip("Run proxy, SLAM, selection, and extraction. Completed stages are skipped.")
        self.sampler_visualize_button.setToolTip("Open the trajectory visualization HTML in the browser.")

    def open_folder(self) -> None:
        folder_text = QFileDialog.getExistingDirectory(self, "Open image folder")
        if not folder_text:
            return
        self.load_folder(Path(folder_text))

    def load_folder(self, folder: Path) -> None:
        if self.mask_dirty and not self.confirm_unsaved_mask():
            return
        self.state.load_folder(folder)
        self.image_list.clear()
        for item in self.state.images:
            list_item = QListWidgetItem(str(item.relative_dir / item.path.name))
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            self.image_list.addItem(list_item)
        self.status_label.setText(f"{len(self.state.images)} images")
        self.refresh_colmap_export_info()
        if self.state.images:
            self.image_list.setCurrentRow(0)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if dropped_path_from_mime(event.mimeData()) is None:
            event.ignore()
            return
        event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        path = dropped_path_from_mime(event.mimeData())
        if path is None:
            event.ignore()
            return
        if is_video_path(path):
            self.import_video(path)
        else:
            self.load_folder(folder_from_path(path))
        event.acceptProposedAction()

    def import_video(self, video_path: Path) -> None:
        if self.mask_dirty and not self.confirm_unsaved_mask():
            return
        fps, ok = QInputDialog.getDouble(
            self,
            "Extract video frames",
            "Output FPS",
            1.0,
            0.01,
            120.0,
            2,
        )
        if not ok:
            return
        quality_select = (
            QMessageBox.question(
                self,
                "Frame selection",
                "Select sharpest frame within each output interval?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            == QMessageBox.StandardButton.Yes
        )

        def task(worker: TaskWorker) -> Path:
            worker.progress.emit(f"Extracting frames: {video_path.name}")
            output_folder = extract_video_frames(video_path, fps, quality_select=quality_select)
            worker.raise_if_cancelled()
            return output_folder

        self.start_worker(task, self.finish_video_import)

    def finish_video_import(self, result: object) -> None:
        if isinstance(result, Path):
            self.load_folder(result)

    def select_image(self, current: QListWidgetItem | None, previous: QListWidgetItem | None = None) -> None:
        if current is None:
            return
        if self._restoring_list_row:
            return
        if self.mask_dirty and not self.confirm_unsaved_mask():
            self._restoring_list_row = True
            try:
                if previous is not None:
                    self.image_list.setCurrentItem(previous)
            finally:
                self._restoring_list_row = False
            return
        item = current.data(Qt.ItemDataRole.UserRole)
        self.current_item = item
        self.current_image = load_rgb(item.path)
        self.current_mask = load_mask(item.mask_path, self.current_image.shape[:2])
        self.mask_dirty = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.refresh_preview()

    def update_brush(self) -> None:
        self.canvas.set_brush(self.brush_spin.value(), self.erase_check.isChecked())

    def refresh_model_list(self) -> None:
        current = self.model_combo.currentData() if hasattr(self, "model_combo") else None
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("Stub / no model", None)
        model_dir = Path(__file__).resolve().parents[3] / "third_party" / "models"
        for path in sorted(model_dir.glob("*.onnx")):
            self.model_combo.addItem(path.name, str(path))
        if current:
            index = self.model_combo.findData(current)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self.load_segmenter()

    def refresh_provider_list(self) -> None:
        current = self.provider_combo.currentData() if hasattr(self, "provider_combo") else None
        self.provider_combo.blockSignals(True)
        self.provider_combo.clear()
        providers = [provider for provider in selectable_onnx_providers() if provider not in self.disabled_provider_names]
        for provider in providers:
            self.provider_combo.addItem(provider_label(provider), provider)
        if current:
            index = self.provider_combo.findData(current)
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)
        self.provider_combo.blockSignals(False)

    def load_segmenter(self) -> bool:
        path_text = self.model_combo.currentData()
        if not path_text:
            self.clear_segmenter()
            return False
        model_path = Path(path_text)
        if not model_path.exists():
            self.clear_segmenter()
            self.status_label.setText(f"Model not found: {model_path}")
            return False
        provider_name = self.provider_combo.currentData()
        try:
            self.load_segmenter_with_provider(model_path, provider_name)
        except Exception as exc:
            self.clear_segmenter()
            if provider_name is not None:
                self.disabled_provider_names.add(provider_name)
                self.refresh_provider_list()
            self.status_label.setText(f"{provider_label(provider_name)} failed: {short_error(exc)}")
            return False
        providers_text = ", ".join(self.segmenter.providers)
        available_text = ", ".join(available_onnx_providers())
        self.status_label.setText(f"Loaded model: {model_path.name} | {providers_text} | available: {available_text}")
        return True

    def clear_segmenter(self) -> None:
        self.segmenter = None
        self.direct_generator = DirectEquirectangularGenerator()
        self._shutdown_cubemap_generator()
        self.cubemap_generator = CubemapGenerator()
        self._update_generate_buttons()

    def load_segmenter_with_provider(self, model_path: Path, provider_name: str | None) -> None:
        providers = resolve_execution_providers(provider_name)
        self.segmenter = DeimWholebodySegmenter(model_path, providers=providers)
        self.direct_generator = DirectEquirectangularGenerator(self.segmenter)
        self._shutdown_cubemap_generator()
        self.cubemap_generator = PersistentCubemapGenerator(model_path, provider_name)
        self._update_generate_buttons()

    def _shutdown_cubemap_generator(self) -> None:
        shutdown = getattr(self.cubemap_generator, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def refresh_preview(self) -> None:
        if self.current_image is None or self.current_mask is None:
            return
        image, mask, scale = self._preview_inputs()
        preview = overlay_mask(image, mask, opacity=self.overlay_opacity_spin.value() / 100.0)
        self.canvas.set_preview(preview, self.current_mask)
        self.status_label.setText(f"Mask area: {mask_area(self.current_mask)} px")

    def schedule_preview_refresh(self) -> None:
        self.preview_timer.start(25)

    def _preview_inputs(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self.current_image is None or self.current_mask is None:
            raise RuntimeError("No current image")
        height, width = self.current_image.shape[:2]
        target_side = max(640, min(2048, max(self.canvas.width(), self.canvas.height()) * 2))
        source_side = max(width, height)
        if source_side <= target_side:
            return self.current_image, self.current_mask, 1.0
        scale = target_side / source_side
        preview_w = max(1, round(width * scale))
        preview_h = max(1, round(height * scale))
        image = cv2.resize(self.current_image, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(self.current_mask, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)
        return image, mask, scale

    def begin_mask_edit(self) -> None:
        if self.current_mask is None:
            return
        self.undo_stack.append(self.current_mask.copy())
        self.redo_stack.clear()

    def commit_canvas_stroke(self, stroke_mask: np.ndarray, erase: bool) -> None:
        if self.current_mask is None:
            return
        if erase:
            self.current_mask = np.where(stroke_mask > 0, 0, self.current_mask).astype(np.uint8)
        else:
            self.current_mask = np.maximum(self.current_mask, stroke_mask).astype(np.uint8)
        self.mask_dirty = True
        self.refresh_preview()

    def undo(self) -> None:
        if self.current_mask is None or not self.undo_stack:
            return
        self.redo_stack.append(self.current_mask.copy())
        self.current_mask = self.undo_stack.pop()
        self.mask_dirty = True
        self.refresh_preview()

    def redo(self) -> None:
        if self.current_mask is None or not self.redo_stack:
            return
        self.undo_stack.append(self.current_mask.copy())
        self.current_mask = self.redo_stack.pop()
        self.mask_dirty = True
        self.refresh_preview()

    def _options(self) -> MaskOptions:
        return MaskOptions(
            strategy=self.strategy_combo.currentText(),
            score_threshold=self.score_spin.value() / 100.0,
            mask_threshold=self.mask_threshold_spin.value() / 100.0,
        )

    def generate_selected(self) -> None:
        self._generate_items(self._selected_image_items())

    def generate_all(self) -> None:
        self._generate_items(list(self.state.images))

    def _generate_items(self, items: list[ImageItem]) -> None:
        if not items:
            return
        options = self._options()

        def task(worker: TaskWorker) -> None:
            for idx, item in enumerate(items, start=1):
                worker.raise_if_cancelled()
                label = f"Generating {idx}/{len(items)}: {item.path.name}"
                worker.progress.emit(label)
                image = load_rgb(item.path)
                mask = self._generate_for_item(item, image, options, worker, label)
                save_mask(item.mask_path, mask)
            return None

        self.start_worker(task, self.finish_generate_all)

    def _selected_image_items(self) -> list[ImageItem]:
        selected = sorted(self.image_list.selectedItems(), key=self.image_list.row)
        return [list_item.data(Qt.ItemDataRole.UserRole) for list_item in selected]

    def _generate_for_item(
        self,
        item: ImageItem,
        image: np.ndarray,
        options: MaskOptions,
        worker: TaskWorker | None = None,
        label: str = "",
    ) -> np.ndarray:
        def cubemap_progress(current: int, total: int, face: str) -> None:
            if worker is not None:
                prefix = label or item.path.name
                worker.progress.emit(f"{prefix} | cubemap {current}/{total}: {face}")

        cancel_event = worker.cancel_event if worker is not None else None

        strategy = options.strategy
        if strategy == "direct":
            result = self.direct_generator.generate(image, options)
            save_mask(item.direct_mask_path, result.mask)
            return result.mask
        if strategy == "cubemap":
            result = self.cubemap_generator.generate(
                image, options, progress=cubemap_progress, cancel_event=cancel_event
            )
            save_mask(item.cubemap_mask_path, result.mask)
            return result.mask
        direct = self.direct_generator.generate(image, options)
        if worker is not None:
            worker.raise_if_cancelled()
        cubemap = self.cubemap_generator.generate(
            image, options, progress=cubemap_progress, cancel_event=cancel_event
        )
        save_mask(item.direct_mask_path, direct.mask)
        save_mask(item.cubemap_mask_path, cubemap.mask)
        return merge_compare_masks(direct.mask, cubemap.mask)

    def save_current_mask(self) -> None:
        if self.current_item is None or self.current_mask is None:
            return
        save_mask(self.current_item.mask_path, self.current_mask)
        self.mask_dirty = False
        self.status_label.setText(f"Saved {self.current_item.mask_path.name}")

    def export_selected(self) -> None:
        self._export_items(self._selected_image_items())

    def export_all(self) -> None:
        self._export_items(list(self.state.images))

    def _export_items(self, items: list[ImageItem]) -> None:
        if not items:
            return
        settings = self._export_settings()
        self.start_worker(lambda worker: self._export_worker(worker, items, settings), lambda result: None)

    def _export_settings(self) -> ColmapExportSettings:
        return ColmapExportSettings(tile_size=self.tile_size_spin.value(), fov_deg=float(self.fov_spin.value()))

    def _export_worker(self, worker: TaskWorker, items: list[ImageItem], settings: ColmapExportSettings) -> None:
        for idx, item in enumerate(items, start=1):
            worker.raise_if_cancelled()
            worker.progress.emit(f"Exporting {idx}/{len(items)}: {item.path.name}")
            self._export_item(item, settings)
        if self.state.export_dir is not None:
            write_colmap_metadata(self.state.export_dir, settings)
            self.refresh_colmap_export_info()
        return None

    def finish_generate_all(self, result: object) -> None:
        if self.current_item:
            self.select_image(self.image_list.currentItem())

    def start_worker(self, task, on_finished) -> None:
        if self.worker_thread is not None:
            self.status_label.setText("Task already running")
            return
        self.set_busy(True)
        thread = QThread(self)
        worker = TaskWorker(task)
        worker.moveToThread(thread)
        worker.progress.connect(self.status_label.setText)
        worker.failed.connect(self.on_worker_failed)
        worker.finished.connect(on_finished)
        worker.finished.connect(lambda result: self.on_worker_done())
        worker.cancelled.connect(self.on_worker_cancelled)
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.worker_thread = thread
        self.worker = worker
        thread.start()

    def on_worker_failed(self, message: str) -> None:
        self.status_label.setText(message)
        self.on_worker_done()

    def on_worker_done(self) -> None:
        thread = self.worker_thread
        self.worker = None
        self.worker_thread = None
        self.set_busy(False)
        if thread is not None:
            thread.quit()

    def cancel_current_task(self) -> None:
        if self.sampler_process is not None:
            self._sampler_cancelled = True
            self._kill_sampler_process_tree()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling Sampler...")
            return
        if self.colmap_process is not None:
            self._colmap_cancelled = True
            self.colmap_process.kill()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling COLMAP...")
            return
        if self.worker is None:
            return
        self.worker.cancel()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling...")

    def on_worker_cancelled(self) -> None:
        self.on_worker_done()
        self.status_label.setText("Task cancelled. Finished images were kept.")
        if self.current_item:
            self.select_image(self.image_list.currentItem())

    def _kill_sampler_process_tree(self) -> None:
        process = self.sampler_process
        if process is None:
            return
        pid = process.processId()
        # The sampler child launches run_video_slam.exe; kill() alone orphans it, so
        # tear down the whole tree on Windows (mirrors sampler workflow.PipelineExecutor.cancel).
        if os.name == "nt" and pid:
            subprocess.run(
                ("taskkill", "/PID", str(pid), "/T", "/F"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        process.kill()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in (
            self.open_button,
            self.model_combo,
            self.provider_combo,
            self.model_button,
            self.export_selected_button,
            self.export_all_button,
            self.run_colmap_button,
            self.align_stella_button,
            self.colmap_browse_button,
            self.colmap_path_edit,
            self.colmap_matcher_combo,
            self.colmap_sparse_mapper_combo,
            self.colmap_overwrite_check,
            self.colmap_skip_completed_check,
            self.colmap_skip_mapping_check,
            self.colmap_rig_ba_check,
            self.colmap_dense_check,
            self.colmap_use_gpu_check,
            self.colmap_gpu_index_edit,
            self.colmap_snapshot_check,
            self.colmap_snapshot_freq_spin,
            self.run_sampler_button,
            self.sampler_video_edit,
            self.sampler_video_browse_button,
            self.sampler_output_edit,
            self.sampler_output_browse_button,
            self.sampler_frame_skip_spin,
            self.sampler_format_combo,
            self.sampler_jpeg_quality_spin,
            self.sampler_exe_edit,
            self.sampler_exe_browse_button,
            self.sampler_vocab_edit,
            self.sampler_vocab_browse_button,
            self.sampler_camera_edit,
            self.sampler_camera_browse_button,
            self.sampler_config_edit,
            self.sampler_config_browse_button,
        ):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self._update_generate_buttons()

    def _update_generate_buttons(self) -> None:
        enabled = not self._busy and self.segmenter is not None
        if self.segmenter is None:
            tooltip_selected = "Load an ONNX model to enable mask generation."
            tooltip_all = tooltip_selected
        else:
            tooltip_selected = "Generate and save masks for the selected images."
            tooltip_all = "Generate and save masks for all images in the folder."
        self.generate_selected_button.setEnabled(enabled)
        self.generate_selected_button.setToolTip(tooltip_selected)
        self.generate_all_button.setEnabled(enabled)
        self.generate_all_button.setToolTip(tooltip_all)

    def _export_item(self, item: ImageItem, settings: ColmapExportSettings) -> None:
        if self.state.export_dir is None:
            return
        export_item_for_colmap(item, self.state.export_dir, settings)

    def browse_colmap_executable(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(self, "Select COLMAP executable")
        if path_text:
            self.colmap_path_edit.setText(path_text)

    def refresh_colmap_export_info(self) -> None:
        if self.state.export_dir is None:
            self.colmap_image_path_label.setText("-")
            self.colmap_image_count_label.setText("0")
            self.colmap_mask_count_label.setText("0")
            self.colmap_registered_count_label.setText("0")
            self.update_colmap_status_panel(None)
            return
        images_dir = self.state.export_dir / "images"
        masks_dir = self.state.export_dir / "masks"
        image_count, mask_count = colmap_image_mask_counts(images_dir, masks_dir)
        registered_count = colmap_registered_image_count(self.state.export_dir)
        self.colmap_image_path_label.setText(str(images_dir))
        self.colmap_image_count_label.setText(str(image_count))
        self.colmap_mask_count_label.setText(str(mask_count))
        self.colmap_registered_count_label.setText(f"{registered_count} / {image_count}")
        running_step = None
        if self.colmap_process is not None and self.colmap_step_index < len(self.colmap_steps):
            running_step = self.colmap_steps[self.colmap_step_index].name
        self.update_colmap_status_panel(
            colmap_pipeline_status(
                self.state.export_dir,
                rig_ba_enabled=self.colmap_rig_ba_check.isChecked(),
                dense_enabled=self.colmap_dense_check.isChecked(),
            ),
            running_step=running_step,
        )

    def update_colmap_status_panel(self, status: ColmapPipelineStatus | None, running_step: str | None = None) -> None:
        if status is None:
            for label in (
                self.colmap_resume_label,
                self.colmap_run_start_label,
                self.colmap_export_status_label,
                self.colmap_feature_status_label,
                self.colmap_rig_status_label,
                self.colmap_match_status_label,
                self.colmap_sparse_status_label,
                self.colmap_rig_ba_status_label,
                self.colmap_dense_status_label,
                self.colmap_snapshot_status_label,
            ):
                label.setText("-")
            return
        self.colmap_resume_label.setText(status.next_step)
        self.colmap_run_start_label.setText(colmap_run_start_text(status, self.colmap_overwrite_check.isChecked(), self.colmap_skip_completed_check.isChecked()))
        self.colmap_export_status_label.setText(status.export_text)
        self.colmap_feature_status_label.setText(step_status_text(status.feature_done, running_step == "Feature extraction"))
        self.colmap_rig_status_label.setText(step_status_text(status.rig_done, running_step == "Rig configuration"))
        self.colmap_match_status_label.setText(
            step_status_text(
                status.matching_done,
                running_step in ("Pair-list matching", "Sequential matching", "Exhaustive matching", "Vocab_Tree matching"),
            )
        )
        self.colmap_sparse_status_label.setText(
            f"{step_status_text(status.sparse_done, running_step in ('Sparse mapping', 'Hierarchical sparse mapping'))} | {status.sparse_registered_count} registered"
        )
        self.colmap_rig_ba_status_label.setText(
            f"{step_status_text(status.rig_ba_done, running_step == 'Rig bundle adjustment')} | {status.rig_ba_registered_count} registered"
        )
        self.colmap_dense_status_label.setText(
            step_status_text(
                status.dense_done,
                running_step in ("Image undistortion", "Patch-match stereo", "Stereo fusion"),
            )
        )
        self.colmap_snapshot_status_label.setText(
            f"{status.snapshot_count} snapshots | latest {status.snapshot_registered_count} registered"
        )

    def align_colmap_to_stella_up(self) -> None:
        if self.state.export_dir is None:
            QMessageBox.warning(self, "Stella Align", "Export COLMAP data first.")
            return
        export_dir = self.state.export_dir
        trajectory_path = export_dir.parent / "stella" / "trajectory.csv"
        output_model = export_dir / "sparse_stella_rot" / "0"
        overwrite = False
        if output_model.exists():
            reply = QMessageBox.question(
                self,
                "Stella Align",
                f"Replace existing aligned model?\n{output_model}",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        try:
            report = align_colmap_model_to_stella_up(export_dir, trajectory_path, output_model, overwrite=overwrite)
        except Exception as exc:
            QMessageBox.warning(self, "Stella Align", str(exc))
            return
        self.append_colmap_log(
            f"Stella up alignment wrote {report.output_model} | matched {report.image_pairs} images | angle {report.angle_deg:.3f} deg"
        )
        self.status_label.setText("Stella alignment finished")
        self.refresh_colmap_export_info()

    def run_colmap_gui(self) -> None:
        if self.colmap_process is not None or self.worker_thread is not None:
            self.status_label.setText("Task already running")
            return
        if self.state.export_dir is None:
            self.status_label.setText("No export folder")
            return
        export_dir = self.state.export_dir
        self.refresh_colmap_export_info()
        try:
            validate_export_dir(export_dir)
            self.prepare_colmap_output(export_dir)
        except (Exception, SystemExit) as exc:
            self.status_label.setText(short_error(exc))
            QMessageBox.warning(self, "COLMAP", str(exc))
            return
        colmap_path = self.colmap_path_edit.text().strip() or "colmap"
        self.colmap_log.clear()
        gpu_options = detect_colmap_gpu_options(colmap_path)
        sparse_mapper = self.colmap_sparse_mapper_combo.currentText()
        mapper_options = detect_colmap_mapper_options(colmap_path, sparse_mapper)
        if self.colmap_use_gpu_check.isChecked() and not (gpu_options.extraction_supported and gpu_options.matching_supported):
            self.append_colmap_log("GPU options unsupported by this COLMAP binary. Running without GPU flags.")
        if self.colmap_snapshot_check.isChecked() and not mapper_options.snapshot_supported:
            self.append_colmap_log("Mapper snapshot options unsupported by this COLMAP binary. Running without snapshots.")
        settings = ColmapRunSettings(
            colmap=colmap_path,
            tile_size=self.tile_size_spin.value(),
            fov_deg=float(self.fov_spin.value()),
            matcher=self.colmap_matcher_combo.currentText(),
            sparse_mapper=sparse_mapper,
            skip_mapping=self.colmap_skip_mapping_check.isChecked(),
            rig_bundle_adjustment=self.colmap_rig_ba_check.isChecked(),
            dense_reconstruction=self.colmap_dense_check.isChecked(),
            skip_completed=self.colmap_skip_completed_check.isChecked(),
            use_gpu=self.colmap_use_gpu_check.isChecked(),
            gpu_index=self.colmap_gpu_index_edit.text().strip() or "-1",
            mapper_snapshot_path=export_dir / "snapshots" if self.colmap_snapshot_check.isChecked() else None,
            mapper_snapshot_images_freq=self.colmap_snapshot_freq_spin.value(),
            gpu_options=gpu_options,
            mapper_options=mapper_options,
        )
        self.colmap_steps = build_colmap_steps(export_dir, settings)
        self.colmap_step_index = 0
        self._colmap_cancelled = False
        self.colmap_progress.setRange(0, len(self.colmap_steps))
        self.colmap_progress.setValue(0)
        if not self.colmap_steps:
            self.colmap_stage_label.setText("Done")
            self.status_label.setText("COLMAP already complete")
            self.append_colmap_log("All COLMAP steps already have output. Nothing to run.")
            return
        self.set_busy(True)
        self.colmap_status_timer.start()
        self.start_next_colmap_step()

    def prepare_colmap_output(self, export_dir: Path) -> None:
        database_path = export_dir / "database.db"
        sparse_dir = export_dir / "sparse"
        rig_ba_dir = export_dir / "sparse_rig_ba"
        dense_dir = export_dir / "dense"
        snapshot_dir = export_dir / "snapshots"
        if should_overwrite_outputs(self.colmap_overwrite_check.isChecked(), self.colmap_skip_completed_check.isChecked()):
            if database_path.exists():
                database_path.unlink()
            if sparse_dir.exists():
                shutil.rmtree(sparse_dir)
            if self.colmap_rig_ba_check.isChecked() and rig_ba_dir.exists():
                shutil.rmtree(rig_ba_dir)
            if self.colmap_dense_check.isChecked() and dense_dir.exists():
                shutil.rmtree(dense_dir)
            if self.colmap_snapshot_check.isChecked() and snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        if self.colmap_snapshot_check.isChecked():
            snapshot_dir.mkdir(parents=True, exist_ok=True)

    def start_next_colmap_step(self) -> None:
        if self.colmap_step_index >= len(self.colmap_steps):
            self.colmap_process = None
            self.set_busy(False)
            self.colmap_status_timer.stop()
            self.refresh_colmap_export_info()
            self.colmap_stage_label.setText("Done")
            self.status_label.setText("COLMAP finished")
            self.append_colmap_log("COLMAP finished")
            return
        step = self.colmap_steps[self.colmap_step_index]
        self.colmap_stage_label.setText(f"{self.colmap_step_index + 1}/{len(self.colmap_steps)} {step.name}")
        self.status_label.setText(step.name)
        self.append_colmap_log(f"> {' '.join(step.command)}")
        process = QProcess(self)
        process.setWorkingDirectory(str(self.state.export_dir))
        process.readyReadStandardOutput.connect(self.read_colmap_stdout)
        process.readyReadStandardError.connect(self.read_colmap_stderr)
        process.finished.connect(self.on_colmap_finished)
        process.errorOccurred.connect(self.on_colmap_error)
        self.colmap_process = process
        process.start(step.command[0], step.command[1:])

    def read_colmap_stdout(self) -> None:
        if self.colmap_process is None:
            return
        self.append_colmap_log(bytes(self.colmap_process.readAllStandardOutput()).decode(errors="replace").rstrip())

    def read_colmap_stderr(self) -> None:
        if self.colmap_process is None:
            return
        self.append_colmap_log(bytes(self.colmap_process.readAllStandardError()).decode(errors="replace").rstrip())

    def append_colmap_log(self, text: str) -> None:
        if text:
            self.colmap_log.appendPlainText(text)

    def on_colmap_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self.colmap_process is None:
            return
        if self._colmap_cancelled:
            self.colmap_process = None
            self.set_busy(False)
            self.colmap_status_timer.stop()
            self.refresh_colmap_export_info()
            self.colmap_stage_label.setText("Cancelled")
            self.status_label.setText("COLMAP cancelled")
            return
        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            self.colmap_process = None
            self.set_busy(False)
            self.colmap_status_timer.stop()
            self.refresh_colmap_export_info()
            self.colmap_stage_label.setText("Failed")
            self.status_label.setText(f"COLMAP failed: exit {exit_code}")
            return
        self.colmap_step_index += 1
        self.colmap_progress.setValue(self.colmap_step_index)
        self.colmap_process = None
        self.refresh_colmap_export_info()
        self.start_next_colmap_step()

    def on_colmap_error(self, error: QProcess.ProcessError) -> None:
        self.append_colmap_log(f"COLMAP process error: {error.name}")
        if error == QProcess.ProcessError.FailedToStart:
            self.colmap_process = None
            self.set_busy(False)
            self.colmap_status_timer.stop()
            self.refresh_colmap_export_info()
            self.colmap_stage_label.setText("Failed")
            self.status_label.setText("COLMAP failed to start")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker_thread is not None or self.colmap_process is not None or self.sampler_process is not None:
            self.status_label.setText("Task running")
            event.ignore()
            return
        if not self.confirm_unsaved_mask():
            event.ignore()
            return
        self._shutdown_cubemap_generator()
        event.accept()

    def confirm_unsaved_mask(self) -> bool:
        if not self.mask_dirty or self.current_item is None or self.current_mask is None:
            return True
        reply = QMessageBox.question(
            self,
            "Save mask",
            "Save current mask changes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Yes:
            self.save_current_mask()
        else:
            self.mask_dirty = False
        return True


def short_error(exc: Exception, limit: int = 180) -> str:
    message = str(exc).replace("\n", " ").strip()
    return message if len(message) <= limit else f"{message[:limit]}..."


def folder_from_mime(mime_data) -> Path | None:
    path = dropped_path_from_mime(mime_data)
    if path is None or is_video_path(path):
        return None
    return folder_from_path(path)


def folder_from_path(path: Path) -> Path:
    if path.is_dir():
        return path
    return path.parent


def colmap_image_mask_counts(images_dir: Path, masks_dir: Path) -> tuple[int, int]:
    if not images_dir.exists():
        return 0, 0
    image_paths = sorted(
        path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    mask_count = 0
    for image_path in image_paths:
        mask_path = masks_dir / image_path.relative_to(images_dir).with_name(f"{image_path.name}.png")
        if mask_path.exists():
            mask_count += 1
    return len(image_paths), mask_count


@dataclass(frozen=True)
class ColmapPipelineStatus:
    image_count: int
    mask_count: int
    export_ready: bool
    feature_done: bool
    rig_done: bool
    matching_done: bool
    sparse_done: bool
    rig_ba_done: bool
    dense_done: bool
    sparse_registered_count: int
    rig_ba_registered_count: int
    snapshot_count: int
    snapshot_registered_count: int
    next_step: str
    export_text: str


def colmap_pipeline_status(
    export_dir: Path,
    rig_ba_enabled: bool = False,
    dense_enabled: bool = False,
) -> ColmapPipelineStatus:
    images_dir = export_dir / "images"
    masks_dir = export_dir / "masks"
    database_path = export_dir / "database.db"
    sparse_dir = export_dir / "sparse"
    rig_ba_dir = export_dir / "sparse_rig_ba"
    dense_dir = export_dir / "dense"
    snapshots_dir = export_dir / "snapshots"
    image_count, mask_count = colmap_image_mask_counts(images_dir, masks_dir)
    export_ready = image_count > 0 and masks_dir.exists() and (export_dir / "rig_config.json").exists()
    feature_done = feature_extraction_done(database_path, image_count)
    rig_done = database_has_rows(database_path, "frames")
    matching_done = database_has_rows(database_path, "matches")
    sparse_done = sparse_model_exists(sparse_dir)
    rig_ba_done = sparse_model_exists(rig_ba_dir)
    dense_done = dense_model_exists(dense_dir)
    sparse_registered_count = colmap_registered_image_count_in_root(sparse_dir)
    rig_ba_registered_count = colmap_registered_image_count_in_root(rig_ba_dir)
    snapshot_model_dirs = colmap_model_dirs(snapshots_dir)
    snapshot_registered_count = colmap_registered_image_count_in_root(snapshots_dir)
    next_step = next_colmap_step(
        export_ready=export_ready,
        feature_done=feature_done,
        rig_done=rig_done,
        matching_done=matching_done,
        sparse_done=sparse_done,
        rig_ba_done=rig_ba_done,
        dense_done=dense_done,
        rig_ba_enabled=rig_ba_enabled,
        dense_enabled=dense_enabled,
    )
    export_text = "ready" if export_ready else "missing"
    export_text = f"{export_text} | {image_count} images, {mask_count} masks"
    return ColmapPipelineStatus(
        image_count=image_count,
        mask_count=mask_count,
        export_ready=export_ready,
        feature_done=feature_done,
        rig_done=rig_done,
        matching_done=matching_done,
        sparse_done=sparse_done,
        rig_ba_done=rig_ba_done,
        dense_done=dense_done,
        sparse_registered_count=sparse_registered_count,
        rig_ba_registered_count=rig_ba_registered_count,
        snapshot_count=len(snapshot_model_dirs),
        snapshot_registered_count=snapshot_registered_count,
        next_step=next_step,
        export_text=export_text,
    )


def next_colmap_step(
    *,
    export_ready: bool,
    feature_done: bool,
    rig_done: bool,
    matching_done: bool,
    sparse_done: bool,
    rig_ba_done: bool,
    dense_done: bool,
    rig_ba_enabled: bool,
    dense_enabled: bool,
) -> str:
    if not export_ready:
        return "Export COLMAP data"
    if not feature_done:
        return "Feature extraction"
    if not rig_done:
        return "Rig configuration"
    if not matching_done:
        return "Feature matching"
    if not sparse_done:
        return "Sparse mapping"
    if rig_ba_enabled and not rig_ba_done:
        return "Rig bundle adjustment"
    if dense_enabled and not dense_done:
        return "Dense reconstruction"
    return "Complete"


def colmap_run_start_text(status: ColmapPipelineStatus, overwrite: bool, skip_completed: bool) -> str:
    if not status.export_ready:
        return "Export required"
    if skip_completed:
        return status.next_step
    if overwrite:
        return "Feature extraction (overwrite)"
    return "Feature extraction (no skip)"


def step_status_text(done: bool, running: bool = False) -> str:
    if running:
        return "running"
    return "done" if done else "pending"


def colmap_registered_image_count(export_dir: Path) -> int:
    model_dir = latest_colmap_model_dir(export_dir)
    if model_dir is None:
        return 0
    return registered_images_in_model(model_dir)


def colmap_registered_image_count_in_root(root: Path) -> int:
    model_dirs = colmap_model_dirs(root)
    if not model_dirs:
        return 0
    return registered_images_in_model(max(model_dirs, key=colmap_model_mtime))


def latest_colmap_model_dir(export_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for root in (export_dir / "sparse", export_dir / "snapshots"):
        candidates.extend(colmap_model_dirs(root))
    if not candidates:
        return None
    return max(candidates, key=colmap_model_mtime)


def colmap_model_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    return [path for path in paths if (path / "images.bin").exists() or (path / "images.txt").exists()]


def colmap_model_mtime(model_dir: Path) -> float:
    for name in ("images.bin", "images.txt"):
        path = model_dir / name
        if path.exists():
            return path.stat().st_mtime
    return 0.0


def registered_images_in_model(model_dir: Path) -> int:
    images_bin = model_dir / "images.bin"
    if images_bin.exists():
        data = images_bin.read_bytes()[:8]
        if len(data) == 8:
            return int(struct.unpack("<Q", data)[0])
        return 0
    images_txt = model_dir / "images.txt"
    if images_txt.exists():
        lines = [
            line
            for line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return len(lines) // 2
    return 0


def default_colmap_executable() -> str:
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "third_party" / "colmap" / "bin" / "colmap.exe",
        project_root / "third_party" / "colmap" / "bin" / "colmap",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "colmap"


def default_sampler_executable() -> str:
    candidate = sampler_project_root() / "third_party" / "runtime" / "run_video_slam.exe"
    return str(candidate)


def default_sampler_vocabulary() -> str:
    candidate = sampler_project_root() / "third_party" / "FBoW_orb_vocab" / "orb_vocab.fbow"
    return str(candidate)


def dropped_path_from_mime(mime_data) -> Path | None:
    if not mime_data.hasUrls():
        return None
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        path = Path(url.toLocalFile())
        if path.is_dir():
            return path
        if path.is_file() and (path.suffix.lower() in IMAGE_EXTENSIONS or is_video_path(path)):
            return path
    return None

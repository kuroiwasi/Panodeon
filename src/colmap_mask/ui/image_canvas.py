from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget


def rgb_to_pixmap(image: np.ndarray) -> QPixmap:
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    qimage = QImage(image.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class ImageCanvas(QWidget):
    strokeStarted = Signal()
    strokeCommitted = Signal(object, bool)

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.preview: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.pixmap: QPixmap | None = None
        self._draft_pixmap: QPixmap | None = None
        self.brush_radius = 24
        self.erase = False
        self._image_rect = QRectF()
        self._last_image_pos: tuple[int, int] | None = None
        self._last_widget_pos: QPoint | None = None
        self._cursor_widget_pos: QPoint | None = None
        self._stroke_points: list[tuple[int, int]] = []
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

    def set_preview(self, preview: np.ndarray, mask: np.ndarray) -> None:
        self.preview = preview
        self.mask = mask
        self._clear_draft()
        self.pixmap = rgb_to_pixmap(preview)
        self.update()

    def set_brush(self, radius: int, erase: bool) -> None:
        self.brush_radius = max(1, radius)
        self.erase = erase

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#1d1d1d"))
        if self.pixmap is None:
            painter.setPen(QColor("#808080"))
            font = painter.font()
            font.setPointSize(13)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Drag & Drop Folder / Images Here\nor Click 'Open Folder'",
            )
            return
        target = self._fit_rect(self.pixmap.width(), self.pixmap.height())
        self._image_rect = target
        painter.fillRect(target, Qt.GlobalColor.black)
        painter.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))
        if self._draft_pixmap is not None:
            painter.drawPixmap(0, 0, self._draft_pixmap)
        self._draw_brush_cursor(painter)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._update_cursor(event.pos())
            pos = self._widget_to_image(event.pos())
            if pos is None:
                return
            self.strokeStarted.emit()
            self._begin_stroke(event.pos(), pos)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._update_cursor(event.pos())
        if event.buttons() & Qt.MouseButton.LeftButton:
            pos = self._widget_to_image(event.pos())
            if pos is None:
                return
            self._append_stroke(event.pos(), pos)
            self._last_image_pos = pos

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mask is not None and self._stroke_points:
                stroke_mask = build_stroke_mask(self.mask.shape[:2], self._stroke_points, self.brush_radius)
                self.strokeCommitted.emit(stroke_mask, self.erase)
                self._clear_draft()
                self.update()
            self._last_image_pos = None
            self._last_widget_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self.pixmap is None:
            return
        cursor_f = event.position()
        cursor = cursor_f.toPoint()
        old_rect = self._fit_rect(self.pixmap.width(), self.pixmap.height())
        self._image_rect = old_rect
        if not old_rect.contains(cursor_f):
            return
        old_scale = old_rect.width() / self.pixmap.width()
        image_x = (cursor_f.x() - old_rect.x()) / old_scale
        image_y = (cursor_f.y() - old_rect.y()) / old_scale
        old_zoom = self.zoom
        steps = event.angleDelta().y() / 120.0
        self.zoom = float(np.clip(self.zoom * (1.2 ** steps), 1.0, 32.0))
        if self.zoom == old_zoom:
            return
        base_scale = self._base_scale(self.pixmap.width(), self.pixmap.height())
        if self.zoom <= 1.0:
            self._center_image(self.pixmap.width(), self.pixmap.height(), base_scale)
            self._image_rect = self._fit_rect(self.pixmap.width(), self.pixmap.height())
            self._clear_draft()
            self.update()
            self._update_cursor(cursor)
            event.accept()
            return
        new_scale = base_scale * self.zoom
        new_w = self.pixmap.width() * new_scale
        new_h = self.pixmap.height() * new_scale
        self.offset_x = cursor_f.x() - image_x * new_scale
        self.offset_y = cursor_f.y() - image_y * new_scale
        self._image_rect = QRectF(
            self.offset_x,
            self.offset_y,
            new_w,
            new_h,
        )
        self._clear_draft()
        self.update()
        self._update_cursor(cursor)
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._cursor_widget_pos = None
        self.update()

    def _fit_rect(self, image_w: int, image_h: int) -> QRectF:
        if image_w <= 0 or image_h <= 0:
            return QRectF()
        scale = self._base_scale(image_w, image_h) * self.zoom
        w = max(1.0, image_w * scale)
        h = max(1.0, image_h * scale)
        if self.zoom <= 1.0:
            self._center_image(image_w, image_h, scale)
        return QRectF(self.offset_x, self.offset_y, w, h)

    def _base_scale(self, image_w: int, image_h: int) -> float:
        if image_w <= 0 or image_h <= 0:
            return 1.0
        return min(self.width() / image_w, self.height() / image_h)

    def _center_image(self, image_w: int, image_h: int, scale: float) -> None:
        self.offset_x = (self.width() - image_w * scale) / 2.0
        self.offset_y = (self.height() - image_h * scale) / 2.0

    def _widget_to_image(self, pos: QPoint) -> tuple[int, int] | None:
        if self.mask is None or self.preview is None or not self._image_rect.contains(pos):
            return None
        h, w = self.mask.shape[:2]
        x = int((pos.x() - self._image_rect.x()) / self._image_rect.width() * w)
        y = int((pos.y() - self._image_rect.y()) / self._image_rect.height() * h)
        return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

    def _update_cursor(self, pos: QPoint) -> None:
        old_pos = self._cursor_widget_pos
        self._cursor_widget_pos = QPoint(pos) if self._image_rect.contains(pos) else None
        self._update_cursor_dirty(old_pos)
        self._update_cursor_dirty(self._cursor_widget_pos)

    def _update_cursor_dirty(self, pos: QPoint | None) -> None:
        if pos is None:
            return
        width = max(4, self._display_brush_width())
        rect = QRect(pos.x() - width, pos.y() - width, width * 2, width * 2)
        self.update(rect.adjusted(-2, -2, 2, 2))

    def _begin_stroke(self, widget_pos: QPoint, image_pos: tuple[int, int]) -> None:
        self._clear_draft()
        self._ensure_draft_pixmap()
        self._stroke_points = [image_pos]
        self._last_image_pos = image_pos
        self._last_widget_pos = QPoint(widget_pos)
        self._draw_overlay_point(widget_pos)

    def _append_stroke(self, widget_pos: QPoint, image_pos: tuple[int, int]) -> None:
        if self._last_widget_pos is None:
            self._begin_stroke(widget_pos, image_pos)
            return
        self._stroke_points.append(image_pos)
        self._draw_overlay_line(self._last_widget_pos, widget_pos)
        self._last_widget_pos = QPoint(widget_pos)

    def _ensure_draft_pixmap(self) -> None:
        if self._draft_pixmap is None or self._draft_pixmap.size() != self.size():
            self._draft_pixmap = QPixmap(self.size())
            self._draft_pixmap.fill(Qt.GlobalColor.transparent)

    def _clear_draft(self) -> None:
        self._draft_pixmap = None
        self._stroke_points = []

    def _display_brush_width(self) -> int:
        if self.mask is None or self._image_rect.width() <= 0:
            return max(1, self.brush_radius * 2)
        scale = self._image_rect.width() / max(1, self.mask.shape[1])
        return max(1, round(self.brush_radius * 2 * scale))

    def _overlay_pen(self) -> QPen:
        color = QColor(120, 163, 200, 170) if self.erase else QColor(229, 125, 34, 170)
        return QPen(
            color,
            self._display_brush_width(),
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )

    def _draw_brush_cursor(self, painter: QPainter) -> None:
        if self._cursor_widget_pos is None or not self._image_rect.contains(self._cursor_widget_pos):
            return
        width = self._display_brush_width()
        radius = max(1, width // 2)
        color = QColor(120, 163, 200, 220) if self.erase else QColor(229, 125, 34, 220)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(self._cursor_widget_pos, radius, radius)

    def _draw_overlay_point(self, point: QPoint) -> None:
        self._ensure_draft_pixmap()
        width = self._display_brush_width()
        painter = QPainter(self._draft_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self._overlay_pen())
        painter.setBrush(self._overlay_pen().color())
        radius = max(1, width // 2)
        painter.drawEllipse(point, radius, radius)
        painter.end()
        self.update(QRect(point.x() - width, point.y() - width, width * 2, width * 2))

    def _draw_overlay_line(self, start: QPoint, end: QPoint) -> None:
        self._ensure_draft_pixmap()
        width = self._display_brush_width()
        painter = QPainter(self._draft_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self._overlay_pen())
        painter.drawLine(start, end)
        painter.end()
        dirty = QRect(start, end).normalized().adjusted(-width, -width, width, width)
        self.update(dirty)


def build_stroke_mask(
    shape: tuple[int, int],
    points: list[tuple[int, int]],
    radius: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not points:
        return mask
    radius = max(1, int(radius))
    if len(points) == 1:
        cv2.circle(mask, points[0], radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        return mask
    thickness = max(1, radius * 2)
    for start, end in zip(points[:-1], points[1:]):
        cv2.line(mask, start, end, 255, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.circle(mask, points[0], radius, 255, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(mask, points[-1], radius, 255, thickness=-1, lineType=cv2.LINE_AA)
    return mask

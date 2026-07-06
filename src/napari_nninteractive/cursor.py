"""Coloured, per-tool mouse cursors for the interaction canvas.

napari's public cursor API (``viewer.cursor.style``) only exposes a fixed set of
named cursors, so it cannot show a green/red per-tool glyph. We instead build a
``QCursor`` from a hand-painted ``QPixmap`` (the same technique the Slicer plugin
uses) and stamp it onto the vispy canvas.

The single choke point napari uses to set the OS cursor is
``VispyCanvas._on_cursor``; it re-runs on every cursor-style, brush-size and zoom
change. Rather than fight those resets with a re-assert timer (as the VTK-backed
Slicer plugin must), we wrap ``_on_cursor``: napari's own logic runs first (so the
brush-size circle overlay for scribble/lasso is preserved), then, while a prompt
tool is active, we stamp our coloured cursor on top. Green means the next prompt is
positive, red means negative; a small badge keeps the four tools visually distinct.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from qtpy.QtCore import Qt
from qtpy.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)

# Match the interaction-layer palette (layers/abstract_layer.py: colors[0]/[1]).
COLOR_POSITIVE = (0.0, 0.694, 0.047)  # green
COLOR_NEGATIVE = (0.827, 0.027, 0.0)  # red

# Tool names keyed by interaction_button.index (options order in widget_gui.py:
# ["Point", "BBox", "Scribble", "Lasso"]).
_TOOL_BY_INDEX = {0: "point", 1: "bbox", 2: "scribble", 3: "lasso"}


def _qcolor(rgb: Tuple[float, float, float], alpha: int = 255) -> QColor:
    return QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), alpha)


def make_prompt_cursor(tool: str, positive: bool) -> QCursor:
    """Build a 32x32 colour cursor: a precise crosshair locator plus a per-tool
    badge, drawn with a dark halo so it reads on both bright and dark images."""
    size = 32
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))  # transparent
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    color = _qcolor(COLOR_POSITIVE if positive else COLOR_NEGATIVE)
    halo = QColor(0, 0, 0, 190)
    cx, cy = 16, 16

    def crosshair(pen_color: QColor, w: int) -> None:
        pen = QPen(pen_color)
        pen.setWidth(w)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(QBrush())
        gap, arm = 3, 7
        p.drawLine(cx - arm, cy, cx - gap, cy)
        p.drawLine(cx + gap, cy, cx + arm, cy)
        p.drawLine(cx, cy - arm, cx, cy - gap)
        p.drawLine(cx, cy + gap, cx, cy + arm)

    def badge(pen_color: QColor, w: int, filled: bool) -> None:
        pen = QPen(pen_color)
        pen.setWidth(w)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(QBrush(pen_color) if filled else QBrush())
        if tool == "point":
            p.drawEllipse(21, 3, 8, 8)
        elif tool == "bbox":
            p.drawRect(21, 4, 8, 8)
        elif tool == "lasso":
            p.drawEllipse(20, 3, 10, 8)
            p.drawLine(25, 11, 28, 14)  # little tail
        elif tool == "scribble":
            # A smooth squiggle drawn as a cubic Bezier chain (never a filled shape).
            path = QPainterPath()
            path.moveTo(19.5, 9.0)
            path.cubicTo(21.0, 3.0, 23.0, 3.0, 24.5, 8.0)
            path.cubicTo(26.0, 13.0, 28.0, 13.0, 29.5, 6.5)
            p.setBrush(QBrush())
            p.drawPath(path)

    # Halo pass (wide, dark) then colour pass (narrow) so the glyph stays legible
    # over any image intensity.
    crosshair(halo, 4)
    badge(halo, 4, False)
    crosshair(color, 2)
    badge(color, 2, tool == "point")
    p.end()
    return QCursor(pm, cx, cy)


class PromptCursorManager:
    """Wrap napari's canvas cursor pipeline to show a coloured per-tool cursor.

    ``state_fn`` is called on every cursor update and must return
    ``(interaction_index, positive)`` where ``interaction_index`` is the
    interaction_button index (or ``None`` when no tool is active) and ``positive``
    is ``True`` for a positive prompt. When no tool is active napari's default
    cursor is left untouched.
    """

    def __init__(
        self, viewer, state_fn: Callable[[], Tuple[Optional[int], bool]]
    ) -> None:
        self._viewer = viewer
        self._state_fn = state_fn
        self._cache: dict = {}
        self._canvas = None
        self._orig_on_cursor: Optional[Callable[[], None]] = None
        self._install()

    def _install(self) -> None:
        try:
            canvas = self._viewer.window._qt_viewer.canvas
        except Exception:  # noqa: BLE001 - private API; degrade gracefully
            return
        # Guard against double-install (e.g. re-init) leaving a nested wrapper.
        if getattr(canvas, "_nnint_cursor_wrapped", False):
            self._canvas = canvas
            self._orig_on_cursor = getattr(canvas, "_nnint_orig_on_cursor", None)
            return
        self._canvas = canvas
        self._orig_on_cursor = canvas._on_cursor
        canvas._nnint_orig_on_cursor = self._orig_on_cursor
        canvas._nnint_cursor_wrapped = True
        canvas._on_cursor = self._on_cursor

    def _get(self, tool: str, positive: bool) -> QCursor:
        key = (tool, bool(positive))
        if key not in self._cache:
            self._cache[key] = make_prompt_cursor(tool, positive)
        return self._cache[key]

    def _on_cursor(self) -> None:
        # napari's own logic first: it manages the brush-size circle overlay for
        # scribble/lasso and the default cursor styles. Never let our override
        # break that pipeline.
        if self._orig_on_cursor is not None:
            self._orig_on_cursor()
        try:
            index, positive = self._state_fn()
        except Exception:  # noqa: BLE001
            return
        tool = _TOOL_BY_INDEX.get(index)
        if tool is None or self._canvas is None:
            return
        # Stamp our coloured cursor on top. For brush tools napari has set a blank
        # cursor plus an in-canvas circle overlay; our badge rides on top of it.
        try:
            self._canvas.cursor = self._get(tool, positive)
        except Exception:  # noqa: BLE001
            pass

    def refresh(self) -> None:
        """Re-evaluate the cursor now (e.g. after a tool or polarity change)."""
        if self._canvas is not None:
            try:
                self._canvas._on_cursor()
            except Exception:  # noqa: BLE001
                pass

    def uninstall(self) -> None:
        """Restore napari's original cursor pipeline."""
        canvas = self._canvas
        if canvas is None:
            return
        if getattr(canvas, "_nnint_cursor_wrapped", False):
            orig = getattr(canvas, "_nnint_orig_on_cursor", None)
            if orig is not None:
                canvas._on_cursor = orig
            for attr in ("_nnint_cursor_wrapped", "_nnint_orig_on_cursor"):
                try:
                    delattr(canvas, attr)
                except AttributeError:
                    pass
            try:
                canvas._on_cursor()  # restore the normal cursor immediately
            except Exception:  # noqa: BLE001
                pass
        self._canvas = None
        self._orig_on_cursor = None

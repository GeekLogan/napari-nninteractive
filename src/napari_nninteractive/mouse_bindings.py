"""Custom mouse controls for the interaction canvas.

While a prompt tool is active, napari would place a prompt on *any* mouse button,
which makes an accidental right-click drop an unwanted prompt. We remap the buttons
so that:

* **left button** places prompts (unchanged),
* **middle button** (press and drag) pans the image,
* **right button** (press and drag up/down) zooms in/out.

Two pieces cooperate:

* ``left_button_only`` wraps napari's edit-mode drag callbacks so only the left
  button edits (see ``layers.abstract_layer``). This frees the other buttons.
* ``MouseControls`` installs viewer-level drag callbacks that drive the camera for
  the middle (pan) and right (zoom) buttons.

Panning works in every mode. Right-drag zoom is only taken over while a prompt tool
is active (an nnInteractive interaction layer is selected); in plain pan/zoom mode
napari's own right-drag zoom is left in place so behaviour is unchanged there.
"""

from __future__ import annotations

import numpy as np

# Vispy button numbers (see vispy.app.backends._qt.BUTTONMAP).
_LEFT, _RIGHT, _MIDDLE = 1, 2, 3

# exp() scaling for right-drag zoom: dragging ~100 px doubles/halves the zoom.
_ZOOM_SENSITIVITY = 0.007

# How many slices Ctrl+Shift+scroll steps per notch (plain Ctrl+scroll steps 1).
_FAST_SCROLL_MULTIPLIER = 5


def left_button_only(callback):
    """Wrap a napari drag callback so it only runs for the left mouse button.

    Works for both plain callbacks and generator (press/move/release) callbacks:
    for a non-left button the wrapper returns ``None`` so napari skips it, leaving
    the right/middle buttons free for pan/zoom.
    """

    def _wrapped(layer, event):
        if getattr(event, "button", _LEFT) != _LEFT:
            return None
        return callback(layer, event)

    # Keep a handle on the original so callers can identify/unwrap if needed.
    _wrapped._nnint_wrapped = callback
    return _wrapped


def _apply_center_delta(viewer, dims_displayed, delta) -> None:
    """Shift camera.center by a world-space delta on the displayed axes only.

    camera.center is always a 3-tuple mapping to the last three world dims, so a
    world dim ``d`` lives at center index ``d - (ndim - 3)``.
    """
    ndim = len(delta)
    offset = ndim - 3
    center = list(viewer.camera.center)
    for d in dims_displayed:
        ci = d - offset
        if 0 <= ci < 3:
            center[ci] += float(delta[d])
    viewer.camera.center = tuple(center)


def _zoom_about(viewer, dims_displayed, anchor, factor) -> None:
    """Scale zoom by ``factor`` while keeping the world point ``anchor`` fixed."""
    viewer.camera.zoom = viewer.camera.zoom * factor
    ndim = len(anchor)
    offset = ndim - 3
    center = list(viewer.camera.center)
    for d in dims_displayed:
        ci = d - offset
        if 0 <= ci < 3:
            center[ci] = float(anchor[d]) - (float(anchor[d]) - center[ci]) / factor
    viewer.camera.center = tuple(center)


def _tool_active(viewer) -> bool:
    """True when an nnInteractive interaction layer is the active layer.

    Those layers force ``mouse_pan = False`` (see ``layers.abstract_layer``), so
    napari's own camera drag is off and we drive pan/zoom ourselves.
    """
    layer = viewer.layers.selection.active
    return getattr(layer, "_nnint_interaction_layer", False)


def pan_with_middle_button(viewer, event):
    """Middle-button drag pans the image (active in every mode)."""
    if event.button != _MIDDLE:
        return
    anchor = np.array(event.position, dtype=float)
    dims_displayed = list(event.dims_displayed)
    event.handled = True
    yield
    while event.type == "mouse_move":
        current = np.array(event.position, dtype=float)
        _apply_center_delta(viewer, dims_displayed, anchor - current)
        event.handled = True
        yield


def zoom_with_right_button(viewer, event):
    """Right-button drag up/down zooms in/out, anchored at the press point.

    Only active while a prompt tool is in use; otherwise napari's built-in
    right-drag zoom handles it.
    """
    if event.button != _RIGHT or not _tool_active(viewer):
        return
    anchor = np.array(event.position, dtype=float)
    dims_displayed = list(event.dims_displayed)
    last_y = event.pos[1]
    event.handled = True
    yield
    while event.type == "mouse_move":
        y = event.pos[1]
        dy = y - last_y
        last_y = y
        # Drag up (dy < 0) zooms in, drag down zooms out.
        factor = float(np.exp(-dy * _ZOOM_SENSITIVITY))
        _zoom_about(viewer, dims_displayed, anchor, factor)
        event.handled = True
        yield


def scroll_slices(viewer, event):
    """Ctrl+scroll steps through slices; Ctrl+Shift+scroll jumps 5 slices at once.

    Replaces napari's built-in ``dims_scroll`` (which only ever steps one slice per
    notch and fires on any Control combo, including Control+Shift). Each notch jumps
    directly to the target slice in a single ``set_current_step`` call (napari clamps
    to the valid range) rather than firing several one-slice increments. The trackpad
    accumulator ``_scroll_progress`` is preserved so smooth scrolling still works.
    """
    if "Control" not in event.modifiers:
        return
    step = _FAST_SCROLL_MULTIPLIER if "Shift" in event.modifiers else 1
    if event.native.inverted():
        viewer.dims._scroll_progress += event.delta[1]
    else:
        viewer.dims._scroll_progress -= event.delta[1]
    while abs(viewer.dims._scroll_progress) >= 1:
        axis = viewer.dims.last_used
        current = viewer.dims.current_step[axis]
        if viewer.dims._scroll_progress < 0:
            viewer.dims.set_current_step(axis, current - step)
            viewer.dims._scroll_progress += 1
        else:
            viewer.dims.set_current_step(axis, current + step)
            viewer.dims._scroll_progress -= 1


class MouseControls:
    """Install/remove the custom pan (middle), zoom (right) and slice-scroll controls."""

    _DRAG_CALLBACKS = (pan_with_middle_button, zoom_with_right_button)

    def __init__(self, viewer) -> None:
        self._viewer = viewer
        self._installed = False
        # napari's built-in dims_scroll callbacks we temporarily replaced.
        self._replaced_wheel_callbacks = []

    def install(self) -> None:
        if self._installed:
            return
        for cb in self._DRAG_CALLBACKS:
            if cb not in self._viewer.mouse_drag_callbacks:
                self._viewer.mouse_drag_callbacks.append(cb)

        # Take over Ctrl(+Shift)+scroll: remove napari's dims_scroll (matched by
        # name so we don't depend on the private import) and install ours, which
        # covers both the 1x and 5x cases.
        self._replaced_wheel_callbacks = [
            cb
            for cb in self._viewer.mouse_wheel_callbacks
            if getattr(cb, "__name__", "") == "dims_scroll"
        ]
        for cb in self._replaced_wheel_callbacks:
            self._viewer.mouse_wheel_callbacks.remove(cb)
        if scroll_slices not in self._viewer.mouse_wheel_callbacks:
            self._viewer.mouse_wheel_callbacks.append(scroll_slices)

        self._installed = True

    def uninstall(self) -> None:
        for cb in self._DRAG_CALLBACKS:
            if cb in self._viewer.mouse_drag_callbacks:
                self._viewer.mouse_drag_callbacks.remove(cb)

        if scroll_slices in self._viewer.mouse_wheel_callbacks:
            self._viewer.mouse_wheel_callbacks.remove(scroll_slices)
        for cb in self._replaced_wheel_callbacks:
            if cb not in self._viewer.mouse_wheel_callbacks:
                self._viewer.mouse_wheel_callbacks.append(cb)
        self._replaced_wheel_callbacks = []

        self._installed = False

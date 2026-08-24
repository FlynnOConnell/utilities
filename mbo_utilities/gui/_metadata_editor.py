"""
Global Metadata Editor Dialog.

This module provides the metadata editor popup that allows users to define
metadata fields before processing or saving data.
"""

from __future__ import annotations

import contextlib
from typing import Any

from imgui_bundle import imgui, hello_imgui

from mbo_utilities.gui._imgui_helpers import PopupAutoSize
from mbo_utilities.metadata import parse_filename_metadata, get_filename_suggestions

_COL_SET = imgui.ImVec4(0.5, 0.8, 0.5, 1.0)
_COL_UNSET = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
_COL_EMPTY = imgui.ImVec4(0.5, 0.5, 0.5, 1.0)
_COL_DETECTED = imgui.ImVec4(0.4, 0.8, 0.9, 1.0)
_COL_CUSTOM_HDR = imgui.ImVec4(0.8, 0.8, 0.2, 1.0)


def _field_tooltip(field: dict) -> str:
    tooltip = field.get("description", "")
    examples = field.get("examples", [])
    if examples:
        tooltip += f"\n\nExamples: {', '.join(examples[:5])}"
    return tooltip


def _input_tooltip(dtype) -> str:
    tip = "Type a value and click Set to save"
    if dtype == str:
        tip += " (text)"
    elif dtype == float:
        tip += " (number)"
    return tip


def _push_wrap_at_edge() -> None:
    # explicit visible-edge wrap; 0.0 would track the scrollable content
    # edge inside a horizontal-scrollbar child
    imgui.push_text_wrap_pos(
        imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
    )


def _apply_set(parent: Any, current_data: Any, canonical: str, dtype, input_key: str) -> None:
    input_val = getattr(parent, input_key).strip()
    if not input_val:
        return
    try:
        parsed = dtype(input_val)
        parent._custom_metadata[canonical] = parsed
        if current_data and hasattr(current_data, "metadata"):
            if isinstance(current_data.metadata, dict):
                current_data.metadata[canonical] = parsed
        setattr(parent, input_key, "")
    except (ValueError, TypeError):
        pass


def _apply_clear(parent: Any, current_data: Any, canonical: str) -> None:
    del parent._custom_metadata[canonical]
    if (current_data and hasattr(current_data, "metadata")
            and isinstance(current_data.metadata, dict)
            and canonical in current_data.metadata):
        del current_data.metadata[canonical]


def _apply_custom_add(parent: Any) -> None:
    val = parent._custom_value
    with contextlib.suppress(ValueError):
        val = float(val) if "." in val else int(val)
    parent._custom_metadata[parent._custom_key.strip()] = val
    parent._custom_key = ""
    parent._custom_value = ""


def _get_suggested_metadata(parent: Any) -> list:
    """Get suggested metadata fields from array."""
    try:
        current_data = parent.image_widget.data[0]
    except (IndexError, AttributeError):
        return []

    fields = []

    # get array-specific suggested fields (e.g., from LBMArray)
    if current_data and hasattr(current_data, "get_suggested_metadata"):
        fields.extend(current_data.get_suggested_metadata())
    # fallback to old name for backwards compat
    elif current_data and hasattr(current_data, "get_required_metadata"):
        fields.extend(current_data.get_required_metadata())

    return fields


def _check_missing_metadata(parent: Any) -> list:
    """Check for missing suggested metadata fields."""
    fields = _get_suggested_metadata(parent)

    missing = []
    for field in fields:
        canonical = field["canonical"]
        custom_val = getattr(parent, "_custom_metadata", {}).get(canonical)
        source_val = field.get("value")
        if custom_val is None and source_val is None:
            missing.append(field)

    return missing


def _build_suggested_fields(parent: Any) -> list[dict]:
    """Build the complete list of suggested metadata fields."""
    # get array data
    try:
        current_data = parent.image_widget.data[0]
    except (IndexError, AttributeError):
        current_data = None

    # get array-specific suggested fields
    suggested_fields = _get_suggested_metadata(parent)
    existing_canonicals = {f["canonical"] for f in suggested_fields}

    # add z-step if not already provided
    z_step_canonicals = ("dz", "z_step_um", "axial_step_um")
    if not any(c in existing_canonicals for c in z_step_canonicals):
        z_step_field = {
            "canonical": "dz",
            "label": "Z Step",
            "unit": "\u03bcm",
            "dtype": float,
            "description": "Distance between Z-planes in micrometers.",
        }
        val = getattr(current_data, "dz", None)
        if val:
            z_step_field["value"] = val
        suggested_fields.append(z_step_field)
        existing_canonicals.add("dz")

    # add frame rate if not already provided (e.g. isoview XML has no fs)
    fs_canonicals = ("fs", "frame_rate", "framerate")
    is_tiled = bool(getattr(current_data, "is_tiled", False))
    if not is_tiled and not any(c in existing_canonicals for c in fs_canonicals):
        fs_field = {
            "canonical": "fs",
            "label": "Frame Rate",
            "unit": "Hz",
            "dtype": float,
            "description": "Volume (or frame) sampling rate in Hz.",
        }
        val = getattr(current_data, "fs", None)
        if val:
            fs_field["value"] = val
        suggested_fields.append(fs_field)
        existing_canonicals.add("fs")

    # parse filename for auto-detected metadata
    filename_meta = None
    if hasattr(parent, "fpath") and parent.fpath:
        fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
        if fpath:
            filename_meta = parse_filename_metadata(str(fpath))

    # add user-provided metadata fields from standard suggestions
    user_fields = get_filename_suggestions()
    for canonical, field_def in user_fields.items():
        if canonical in existing_canonicals:
            continue

        field = dict(field_def)
        # check if value detected from filename
        if filename_meta:
            detected_val = getattr(filename_meta, canonical, None)
            if detected_val:
                field["value"] = detected_val
                field["detected"] = True  # mark as auto-detected

        # check if value in array metadata
        if current_data and hasattr(current_data, "metadata"):
            meta = current_data.metadata
            if isinstance(meta, dict) and canonical in meta:
                field["value"] = meta[canonical]

        suggested_fields.append(field)
        existing_canonicals.add(canonical)

    return suggested_fields


def draw_metadata_editor_content(parent: Any):
    """Draw metadata editor fields. can be embedded in any container (tab, popup, etc.)."""
    if not hasattr(parent, "_custom_metadata"):
        parent._custom_metadata = {}
    if not hasattr(parent, "_custom_key"):
        parent._custom_key = ""
    if not hasattr(parent, "_custom_value"):
        parent._custom_value = ""

    # get array data
    try:
        current_data = parent.image_widget.data[0]
    except (IndexError, AttributeError):
        current_data = None

    # build suggested fields (includes filename detection)
    suggested_fields = _build_suggested_fields(parent)

    table_flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
    # column widths shared by suggested + custom tables so everything aligns
    col_label = hello_imgui.em_size(7)
    col_value = hello_imgui.em_size(10)
    col_input = hello_imgui.em_size(8)
    col_btn = hello_imgui.em_size(6)  # widened to fit Set + delete-X
    input_w = hello_imgui.em_size(7.5)

    # narrow container (docked side panel): stacked rows instead of the
    # fixed-width table, which would clip past the right edge
    table_w = col_label + col_value + col_input + col_btn
    if imgui.get_content_region_avail().x < table_w + hello_imgui.em_size(1):
        _draw_editor_narrow(parent, current_data, suggested_fields)
        return

    # draw suggested fields in a table
    if suggested_fields:
        if imgui.begin_table("suggested_meta", 4, table_flags):
            imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed, col_label)
            imgui.table_setup_column("value", imgui.TableColumnFlags_.width_fixed, col_value)
            imgui.table_setup_column("input", imgui.TableColumnFlags_.width_fixed, col_input)
            imgui.table_setup_column("btn", imgui.TableColumnFlags_.width_fixed, col_btn)

            for field in suggested_fields:
                canonical = field["canonical"]
                label = field["label"]
                unit = field.get("unit", "")
                dtype = field.get("dtype", str)
                detected = field.get("detected", False)

                # get current value (custom overrides source)
                custom_val = parent._custom_metadata.get(canonical)
                source_val = field.get("value")
                value = custom_val if custom_val is not None else source_val
                is_set = value is not None

                imgui.table_next_row()

                # label column
                imgui.table_next_column()
                imgui.text_colored(_COL_SET if is_set else _COL_UNSET, label)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(_field_tooltip(field))

                # value column
                imgui.table_next_column()
                if is_set:
                    val_str = f"{value} {unit}".strip()
                    if detected and custom_val is None:
                        imgui.text_colored(_COL_DETECTED, val_str)
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Detected from filename")
                    else:
                        imgui.text_colored(_COL_SET, val_str)
                else:
                    imgui.text_colored(_COL_EMPTY, "-")

                # input column
                imgui.table_next_column()
                input_key = f"_meta_input_{canonical}"
                if not hasattr(parent, input_key):
                    setattr(parent, input_key, "")

                imgui.set_next_item_width(input_w)
                flags = imgui.InputTextFlags_.chars_decimal if dtype in (float, int) else 0
                _, new_val = imgui.input_text(f"##{canonical}", getattr(parent, input_key), flags=flags)
                setattr(parent, input_key, new_val)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(_input_tooltip(dtype))

                # button column: Set (always) + X delete (only when user has overridden)
                imgui.table_next_column()
                if imgui.small_button(f"Set##{canonical}"):
                    _apply_set(parent, current_data, canonical, dtype, input_key)
                if custom_val is not None:
                    imgui.same_line()
                    if imgui.small_button(f"X##del_{canonical}"):
                        _apply_clear(parent, current_data, canonical)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Clear this override")

            imgui.end_table()

        imgui.spacing()

    # === Custom section ===
    suggested_keys = {f["canonical"] for f in suggested_fields}
    custom_entries = [(k, v) for k, v in parent._custom_metadata.items() if k not in suggested_keys]

    imgui.spacing()
    imgui.separator()
    imgui.text_colored(_COL_CUSTOM_HDR, "Custom")
    imgui.dummy(imgui.ImVec2(0, 2))

    if imgui.begin_table("custom_meta", 4, table_flags):
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed, col_label)
        imgui.table_setup_column("value", imgui.TableColumnFlags_.width_fixed, col_value)
        imgui.table_setup_column("input", imgui.TableColumnFlags_.width_fixed, col_input)
        imgui.table_setup_column("btn", imgui.TableColumnFlags_.width_fixed, col_btn)

        # existing custom entries \u2014 same row layout as suggested table
        to_remove = None
        for key, value in custom_entries:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_colored(_COL_SET, key)
            imgui.table_next_column()
            imgui.text_colored(_COL_SET, str(value))
            imgui.table_next_column()  # input col left blank
            imgui.table_next_column()
            if imgui.small_button(f"X##custom_del_{key}"):
                to_remove = key
            if imgui.is_item_hovered():
                imgui.set_tooltip("Delete this custom entry")
        if to_remove:
            del parent._custom_metadata[to_remove]

        # new-entry row \u2014 key in label col, value in input col, Set in btn col
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.set_next_item_width(col_label)
        _, parent._custom_key = imgui.input_text("##custom_key", parent._custom_key)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Custom key name")

        imgui.table_next_column()  # value col empty until Set
        imgui.table_next_column()
        imgui.set_next_item_width(input_w)
        _, parent._custom_value = imgui.input_text("##custom_val", parent._custom_value)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Custom value (numbers auto-detected)")

        imgui.table_next_column()
        if imgui.small_button("Set##custom_add") and parent._custom_key.strip():
            _apply_custom_add(parent)

        imgui.end_table()


def _draw_editor_narrow(parent: Any, current_data: Any, suggested_fields: list[dict]) -> None:
    """Stacked per-field layout: label/value line, then input + buttons.

    Fits any panel width — text wraps and the input shrinks to leave room
    for its buttons, so nothing crosses the right edge.
    """
    style = imgui.get_style()
    set_w = imgui.calc_text_size("Set").x + style.frame_padding.x * 2
    x_w = imgui.calc_text_size("X").x + style.frame_padding.x * 2

    for field in suggested_fields:
        canonical = field["canonical"]
        dtype = field.get("dtype", str)
        custom_val = parent._custom_metadata.get(canonical)
        source_val = field.get("value")
        value = custom_val if custom_val is not None else source_val
        is_set = value is not None

        _push_wrap_at_edge()
        imgui.text_colored(_COL_SET if is_set else _COL_UNSET, field["label"])
        if imgui.is_item_hovered():
            imgui.set_tooltip(_field_tooltip(field))
        imgui.same_line()
        if is_set:
            val_str = f"{value} {field.get('unit', '')}".strip()
            if field.get("detected", False) and custom_val is None:
                imgui.text_colored(_COL_DETECTED, val_str)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Detected from filename")
            else:
                imgui.text_colored(_COL_SET, val_str)
        else:
            imgui.text_colored(_COL_EMPTY, "-")
        imgui.pop_text_wrap_pos()

        input_key = f"_meta_input_{canonical}"
        if not hasattr(parent, input_key):
            setattr(parent, input_key, "")
        reserve = set_w + style.item_spacing.x
        if custom_val is not None:
            reserve += x_w + style.item_spacing.x
        imgui.set_next_item_width(-reserve)
        flags = imgui.InputTextFlags_.chars_decimal if dtype in (float, int) else 0
        _, new_val = imgui.input_text(
            f"##{canonical}", getattr(parent, input_key), flags=flags
        )
        setattr(parent, input_key, new_val)
        if imgui.is_item_hovered():
            imgui.set_tooltip(_input_tooltip(dtype))
        imgui.same_line()
        if imgui.button(f"Set##{canonical}"):
            _apply_set(parent, current_data, canonical, dtype, input_key)
        if custom_val is not None:
            imgui.same_line()
            if imgui.button(f"X##del_{canonical}"):
                _apply_clear(parent, current_data, canonical)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Clear this override")
        imgui.spacing()

    imgui.spacing()
    imgui.separator()
    imgui.text_colored(_COL_CUSTOM_HDR, "Custom")
    imgui.dummy(imgui.ImVec2(0, 2))

    suggested_keys = {f["canonical"] for f in suggested_fields}
    custom_entries = [
        (k, v) for k, v in parent._custom_metadata.items() if k not in suggested_keys
    ]
    to_remove = None
    for key, value in custom_entries:
        if imgui.small_button(f"X##custom_del_{key}"):
            to_remove = key
        if imgui.is_item_hovered():
            imgui.set_tooltip("Delete this custom entry")
        imgui.same_line()
        _push_wrap_at_edge()
        imgui.text_colored(_COL_SET, f"{key}: {value}")
        imgui.pop_text_wrap_pos()
    if to_remove:
        del parent._custom_metadata[to_remove]

    imgui.set_next_item_width(-1)
    _, parent._custom_key = imgui.input_text_with_hint(
        "##custom_key", "key", parent._custom_key
    )
    if imgui.is_item_hovered():
        imgui.set_tooltip("Custom key name")
    imgui.set_next_item_width(-(set_w + style.item_spacing.x))
    _, parent._custom_value = imgui.input_text_with_hint(
        "##custom_val", "value", parent._custom_value
    )
    if imgui.is_item_hovered():
        imgui.set_tooltip("Custom value (numbers auto-detected)")
    imgui.same_line()
    if imgui.button("Set##custom_add") and parent._custom_key.strip():
        _apply_custom_add(parent)


def draw_metadata_popup(parent: Any) -> None:
    """Standalone "Set Metadata" popup.

    Driven by `parent._show_metadata_popup` (set True from the File menu
    or the Shift+M keybind). Draws every frame; opens the modal once when
    the flag flips on, then resets the flag. The modal stays open until
    the user closes it via the X button, the Close button, or Escape.
    """
    if not hasattr(parent, "_show_metadata_popup"):
        parent._show_metadata_popup = False
    if not hasattr(parent, "_metadata_sizer"):
        parent._metadata_sizer = PopupAutoSize(
            "Set Metadata##MetadataPopup", auto_resize=False
        )

    if parent._show_metadata_popup:
        parent._metadata_sizer.before_open()
        imgui.open_popup("Set Metadata##MetadataPopup")
        parent._show_metadata_popup = False

    # default size; top anchor comes from _metadata_sizer.
    io = imgui.get_io()
    screen_w, screen_h = io.display_size.x, io.display_size.y
    win_w = min(560, screen_w * 0.7)
    win_h = min(620, screen_h * 0.85)
    imgui.set_next_window_size(
        imgui.ImVec2(win_w, win_h), imgui.Cond_.first_use_ever
    )
    imgui.set_next_window_size_constraints(
        imgui.ImVec2(420, 360), imgui.ImVec2(screen_w, screen_h)
    )

    opened, visible = imgui.begin_popup_modal(
        "Set Metadata##MetadataPopup",
        p_open=True,
        flags=imgui.WindowFlags_.no_saved_settings,
    )
    if not opened:
        return
    try:
        if not visible:
            imgui.close_current_popup()
            return
        imgui.dummy(imgui.ImVec2(0, 4))
        # scrollable content area so the close button stays anchored at
        # the bottom no matter how many fields the editor renders.
        avail = imgui.get_content_region_avail()
        content_h = max(0.0, avail.y - 36.0)
        if imgui.begin_child(
            "##MetadataContent",
            imgui.ImVec2(0, content_h),
            imgui.ChildFlags_.borders,
        ):
            draw_metadata_editor_content(parent)
        imgui.end_child()

        imgui.spacing()
        btn_w = 80
        imgui.set_cursor_pos_x((imgui.get_window_width() - btn_w) * 0.5)
        if imgui.button("Close", imgui.ImVec2(btn_w, 0)):
            imgui.close_current_popup()
    finally:
        imgui.end_popup()

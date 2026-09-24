"""Context variant: serialize the user task and tool name in front of the tool output, as plain text."""

from __future__ import annotations

OUTPUT_FIELD = "{tool_output_text}"


def split_template(template: str) -> str:
    """Return the prefix part of the template (everything before the tool output), which must come last."""
    if template.count(OUTPUT_FIELD) != 1 or not template.endswith(OUTPUT_FIELD):
        raise ValueError(f"Context template must end with a single {OUTPUT_FIELD}: {template!r}")
    return template[: -len(OUTPUT_FIELD)]


def build_prefix(template: str, user_task: str, tool_name: str) -> str:
    if not isinstance(user_task, str) or not user_task.strip() or not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError(f"Empty user_task or tool_name ({user_task!r}, {tool_name!r}); the context variant does not invent them")
    return split_template(template).format(user_task=user_task, tool_name=tool_name)


def context_stride(output_room: int, overlap: int) -> int:
    """Stride over output tokens when each window holds `output_room` of them: keep the same token overlap as
    output-only chunking, but never let the stride fall below half a window (so a long prefix can't make stride tiny)."""
    return max(output_room - overlap, max(1, output_room // 2))

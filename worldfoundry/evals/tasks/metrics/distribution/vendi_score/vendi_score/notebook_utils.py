import copy

_display = __import__("IPython.display", fromlist=["Markdown", "display"])
Markdown = _display.Markdown
display = _display.display


def markdown_table(headers, rows, bold_left=True, do_display=True):
    lines = [headers, [":---"] + [":---:" for _ in range(len(headers) - 2)] + ["---:"], *rows]
    if bold_left:
        lines = [copy.copy(row) for row in lines]
        for row in lines[2:]:
            row[0] = f"**{row[0]}**"
    out = "\n".join(["|".join(list(map(str, row))) for row in lines])
    if do_display:
        display(Markdown(out))
    else:
        return out

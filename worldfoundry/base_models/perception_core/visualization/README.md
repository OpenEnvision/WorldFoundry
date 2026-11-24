# `perception_core/visualization`

Rendering helpers that turn perception outputs into images (not a metric).

| Path | What it does | Used by |
| --- | --- | --- |
| `render.py` | Renders depth, normals, flow, boxes, and masks from the official models into preview images | `wf models visualizations` (`worldfoundry/cli/models.py`), Grounding DINO inference |
| `colormaps.py` | Depth and normal colour maps | `render.py` |

## Usage

```bash
wf models visualizations --family perception_core
```

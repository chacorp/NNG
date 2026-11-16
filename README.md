# NNG — Neural Network Graph for Overview Figure
by vibe coding

I made it to make a detailed network architecture figure for the research paper.

<img src="nng_app/assets/capture.png" alt="drawing"/>

A desktop app for designing neural-network architectures as node graphs, drawing them cleanly like a paper overview figure, and exporting them as `torch.nn` code.

- Place layer, activation and operator blocks and connect them. The input→output axis is laid out as one straight top-to-bottom line.
- Group several blocks into a custom block (sub-graph) with Ctrl+G, and edit it in its own tab.
- Export the graph as `torch.nn` code and check it inside the app: forward output shapes and gradient flow of a dummy loss.
- Save the figure as SVG / PNG for papers.

## Run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv run nng
```

The first run creates the environment and installs the dependencies (PySide6, torch). Self-test (exports every example and checks it with torch):

```bash
uv run nng --self-test
```

## Layout

| Path | Contents |
|---|---|
| `ui/index.html`, `ui/style.css` | Viewer / editor page |
| `ui/js/` | Graph model, shape inference, layout, rendering, code generation, examples |
| `nng_app/app.py` | App window, file dialogs, bridge to the page, self-test |
| `nng_app/torch_check.py` | Runs exported code and reports the shape / gradient check |

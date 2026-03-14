# Notebook Writer Conventions

Enforces consistent notebook structure and output conventions across all pipeline notebooks.

---

## Cell Order (per notebook)

1. **Markdown: Stage title** -- stage title, description, inputs consumed
2. **Imports and configuration** -- all imports, pdtoolkit setup, matplotlib config
3. **Data loading** -- read from disk, never re-derive from earlier stages
4. **Analysis cells** -- core computation and plotting
5. **Results cells** -- summary tables, key metrics
6. **Markdown: Summary table** -- key findings, flags, recommended actions (see template below)

---

## Plot Conventions

- Always save plots to disk (use `RUN_DIR` variable set in the notebook):
  ```python
  plt.savefig(f'{RUN_DIR}/figures/XX_description.png', dpi=150, bbox_inches='tight')
  plt.close()
  ```
- **Never use `plt.show()`** -- this embeds output in the notebook
- Standard figure size: `(10, 6)` for single plots, `(14, 6)` for side-by-side
- Colour palette:
  - Blue `#2166AC` -- good / pass / positive
  - Red `#D6604D` -- bad / fail / negative
  - Grey `#999999` -- neutral / reference

---

## Execution Rule

After writing the notebook source, always execute this sequence:

1. **Execute the notebook:**
   ```bash
   jupyter nbconvert --to notebook --execute --inplace {RUN_DIR}/notebooks/XX_name.ipynb
   ```
2. **If execution fails:** read the error, fix the cell, and retry before returning
3. **After successful execution, clear all outputs:**
   ```bash
   jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --inplace {RUN_DIR}/notebooks/XX_name.ipynb
   ```

The notebook on disk is always clean source -- outputs are only in `figures/`.

---

## Summary Cell Format

The last cell of every notebook must be a markdown cell with this exact structure:

```markdown
## Stage Summary

| Item | Value | Status |
|---|---|---|
| Key metric 1 | value | PASS / WARN / FAIL |
| Key metric 2 | value | PASS / WARN / FAIL |

**Flags for human review:** [list of flags or "None"]

**Recommended action for next stage:** [brief instruction]
```

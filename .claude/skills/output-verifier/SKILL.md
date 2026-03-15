# Output Verifier

Checklist to run inline after each subagent completes, before presenting results to the human. Fast structural checks only -- no domain reasoning.

---

## Checklist

Standard checks (apply to all stages including 04a, 04b, 04c):

- [ ] Notebook file exists at expected path
- [ ] nbconvert execution completed without errors (check for `[NbConvertApp] Writing` in output)
- [ ] Stage `.md` file exists and contains all required fields (see each stage definition)
- [ ] `figures/` contains at least one new `.png` for this stage
- [ ] No cell in the notebook has `"output_type": "error"` in its outputs
- [ ] Summary cell is present as the last cell

### Stage 04x — Model Comparator (additional checks)

- [ ] `model_params.json` exists at canonical path (champion copy)
- [ ] `stage_04x.md` contains comparison table with metrics for all completed methods
- [ ] Champion method is declared in `stage_04x.md`
- [ ] At least 1 model builder completed successfully (2+ preferred for meaningful comparison)

---

## On Failure

- Do **not** present to human
- Re-invoke the same subagent with a specific failure message describing which checks failed
- Maximum **2 retries** before surfacing the error to the human for manual intervention

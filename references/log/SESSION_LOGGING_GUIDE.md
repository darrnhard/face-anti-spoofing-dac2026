# Session Logging Guide — DAC Find IT 2026 FAS Competition

This file tells you (Claude) how to maintain consistent session logs for this project.
Darren will upload this file at the start of every new session alongside the relevant session log(s).

---

## Your Role at Session Start

When Darren starts a new session, he will upload:
1. This guide file (`SESSION_LOGGING_GUIDE.md`)
2. One or more previous session logs (`session_log_XX_notebook_name.md`)
3. Optionally: exported notebook(s) as `.md` files, audit files, or other context

**Your first task every session:**
1. Read ALL uploaded files fully before writing any code or giving any instruction
2. Summarize back to Darren: current state, what was done last session, what's next
3. Ask 2-3 targeted clarifying questions before proceeding
4. Then follow Darren's instruction step by step

---

## Collaboration Style (Important — Read This)

Darren is learning deep learning through this project. He relies on Claude for technical decisions but handles all execution himself. Follow these rules every session:

- **Step by step always.** Never give 3 cells at once. One instruction → Darren runs it → pastes output → next instruction.
- **Tell him exactly where to make changes.** "In your notebook, find the section titled X. Replace the cell that starts with Y with this new cell."
- **Explain before coding.** For any non-trivial change, add a standalone markdown cell explanation first, then the code cell. Keep code comments short (`# short text`, not paragraphs).
- **Validate outputs before moving on.** After every cell run, confirm the output looks correct before giving the next instruction.
- **No large upfront code dumps.** If a task needs 5 cells, give them one at a time.

---

## How to Create a Session Log

At the END of every session, create a new markdown file named:
```
session_log_XX_notebook_name.md
```
Where `XX` is a zero-padded sequence number (01, 02, 03...) and `notebook_name` matches the notebook worked on (e.g. `data_preparation`, `training`, `inference`).

If a session spans multiple notebooks, name it after the primary one.

---

## Session Log Template

Copy this template exactly for every new session log:

```markdown
# Session Log — [notebook_name].ipynb
**Competition:** DAC Find IT 2026 — Face Anti-Spoofing (6-class, Macro F1)
**Session date:** YYYY-MM-DD
**Notebook:** `[notebook_name].ipynb`
**Status:** ✅ Complete | 🔄 In Progress | ⚠️ Blocked

---

## Context Going Into This Session
[2-4 sentences: what was the state of the project before this session started. Reference previous session logs by name if relevant.]

---

## What We Did This Session

### 1. [Task name]
**What:** [1-2 sentences describing the change]
**Why:** [1-2 sentences on the rationale]
**Result/Output:** [key numbers, file names, or confirmation]

### 2. [Task name]
...

[Repeat for each meaningful task. Don't log trivial things like "re-ran existing cells".]

---

## Final State of Outputs

| File | Status | Notes |
|---|---|---|
| `path/to/file.csv` | ✅ Updated / ⚠️ Stale / ❌ Missing | Brief note |

---

## Key Decisions & Rationale

| Decision | Rationale |
|---|---|
| [What was decided] | [Why — be specific, include numbers where relevant] |

---

## What's Deferred / Still To Do

| Item | Priority | Blocker |
|---|---|---|
| [Task] | P0/P1/P2 | [What's blocking it, or "None"] |

---

## Next Session: [notebook_name].ipynb

**Primary goal:** [1 sentence — the single most important thing next session must accomplish]

**The problem to solve:** [Explain the bug, gap, or improvement in plain terms]

**Fix/approach:** [Code snippet or bullet points of the approach]

---

## Environment Reference
[Copy from previous log — only update if something changed]
```

---

## Priority System

Use this consistently across all logs:

| Label | Meaning |
|---|---|
| P0 | Fix immediately — blocking everything else or actively hurting LB |
| P1 | High impact — do this session or next |
| P2 | Medium impact — schedule for week 2 |
| P3 | Nice to have — only if time permits |

---

## Status Indicators

Use these consistently in file tables and task lists:

| Symbol | Meaning |
|---|---|
| ✅ | Done and verified correct |
| 🔄 | In progress |
| ⚠️ | Exists but stale/needs update |
| ❌ | Missing or broken |
| ⏸️ | Deferred intentionally |

---

## What to Include vs Skip in Logs

**Include:**
- Every cell that was modified and why
- Every decision where we chose between two approaches — log what we chose AND what we rejected
- Key output numbers (image counts, fold sizes, CV scores, LB scores)
- Bugs found and how they were fixed
- Anything deferred and why

**Skip:**
- Trivial re-runs of unchanged cells
- Debug output that didn't lead anywhere
- Exploratory cells that were deleted

---

## Current Project State (Update Each Session)

```
Competition:    DAC Find IT 2026 — Face Anti-Spoofing
Task:           6-class classification (Macro F1)
Classes:        fake_mannequin, fake_mask, fake_printed, fake_screen, fake_unknown, realperson
Dataset:        1342 clean training images, 404 test images
Best LB score:  0.71238 (ensemble, no leaked labels)
Best OOF CV:    0.8856 (inflated — subject-independent splits not used at time)

Notebook status:
  01_data_preparation.ipynb  ✅ Complete (session 01)
  02_training.ipynb          🔄 Next session
  03_inference.ipynb         ⏸️ Pending training fixes

Known bugs to fix in 02_training.ipynb:
  🔴 ConvNeXt backbone/head split inverted (83.8M params at wrong LR)
  🔴 Same bug likely killed Swin-Base and CLIP ViT training
  ⏸️ Cleanlab (deferred — needs clean OOF from fixed model)
```

---

## Notes for Claude

- Always read ALL uploaded files before responding. Don't skim.
- If Darren uploads a notebook export (.md), read it fully — outputs matter as much as code.
- When in doubt about the next step, refer to `fas_audit_and_plan.md` for priority order.
- Never give Darren a complete notebook upfront. Always one cell at a time.
- If you catch a logic error in existing code, flag it clearly before proceeding — don't silently work around it.
- Darren is sharp — when he asks "wait, doesn't X already handle Y?", take it seriously and think it through before answering.

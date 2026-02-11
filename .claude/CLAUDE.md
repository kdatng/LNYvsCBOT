# CLAUDE.md - Agent Orchestration & Project Guidelines

## Core Principle: Maximum Parallel Agent Spawning

When working on this project, **always spawn as many agents as possible in parallel** to maximize efficiency and accuracy. Every independent task should be its own agent. Do not serialize work that can be parallelized.

### Agent Spawning Rules

1. **Explore agents** for any codebase navigation, file discovery, or context gathering
2. **Bash agents** for independent command execution (installs, builds, tests)
3. **General-purpose agents** for research, multi-step analysis, and complex reasoning
4. **Plan agents** for architectural decisions before implementation

### When to Spawn Multiple Agents

- Data loading AND feature engineering AND model architecture design → 3 parallel agents
- Statistical tests AND deep learning training AND visualization → 3 parallel agents
- File writing AND dependency installation AND verification → 3 parallel agents
- Any time you identify 2+ tasks with no dependency between them → parallel agents

### Project Structure

```
LNYvsCBOT/
├── .claude/
│   └── CLAUDE.md              # This file - agent guidelines
├── skills/
│   └── deep_learning_ref.md   # DL/ML skill reference for all agents
├── lny_dates.json             # LNY holiday dates 2010-2025
├── ZSZMZC_OHLCVOI_2010_2025.CSV  # CBOT futures data (Soybeans, Meal, Corn)
├── lny_cbot_model.py          # Main deep learning analysis script
└── README.md
```

### Data Notes

- **CSV format**: 3 contract groups (Sc1=Soybeans, SMc1=Soybean Meal, Cc1=Corn), reverse chronological, multi-header (skip rows 0-2)
- **JSON format**: LNY dates with official_holiday and extended_window periods
- **Date range**: 2010-01-04 to 2025-05-01 (~3862 trading days)

### Model Development Guidelines

- Always reference `skills/deep_learning_ref.md` for architecture choices
- Use temporal train/val/test splits (never shuffle time series)
- Apply anti-overfitting: dropout, weight decay, early stopping, batch norm
- Statistical significance tests alongside deep learning results
- Design models to accept new data without retraining from scratch

### Anti-Overfitting Checklist (Mandatory)

- [ ] Temporal split (no future leakage)
- [ ] Dropout ≥ 0.2 on dense layers
- [ ] Weight decay ≥ 1e-5
- [ ] Early stopping with patience ≥ 10
- [ ] Learning rate scheduling
- [ ] Gradient clipping ≤ 1.0
- [ ] Validation loss monitoring
- [ ] Cross-validation where sample size permits

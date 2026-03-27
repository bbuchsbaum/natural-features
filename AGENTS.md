## beads_rust (`br`) workflow

This repo uses [`beads_rust`](https://github.com/Dicklesworthstone/beads_rust) for local issue tracking in `.beads/`.

### Core commands

```bash
br ready                              # show unblocked work
br create "<title>" --type task --priority 2
br update <issue-id> --status in_progress
br close <issue-id> --reason "Completed"
br sync --flush-only                  # export DB changes to JSONL
```

### Session checklist

```bash
git add <files>
br sync --flush-only
git add .beads/
git commit -m "sync beads"
```

Notes:
- `br` is non-invasive and never runs git commands for you.
- Use priorities `0-4` (P0 highest, P4 backlog).

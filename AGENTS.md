## mote workflow

This repo uses `mote` for local issue tracking and path coordination.

### Core commands

```bash
mote ready                            # show unblocked open work
mote board                            # compact board overview
mote new "<title>" --priority 2
mote set <issue-id> status=doing
mote show <issue-id>
mote note --kind progress <issue-id> "<progress note>"
mote close <issue-id>
```

### Coordination commands

```bash
mote begin <issue-id> --paths <path>   # claim work and reserve paths
mote reserve --issue <issue-id> <path> # reserve additional paths
mote who-has <path>                    # inspect overlapping reservations
mote done <issue-id>                   # close work and release reservations
```

Notes:
- Use `mote ready` or `mote board` before choosing tracker-backed work.
- Use path reservations for non-trivial edits that may overlap with other agents.
- Use priorities `0-3` (`0` is highest).
- Keep commits scoped to the files changed for the active issue.

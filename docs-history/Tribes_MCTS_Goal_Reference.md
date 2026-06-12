# Tribes MCTS Goal Reference

Working reference for the long-running Java-vs-native parity audit.

## Repo Root

`C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\Tribes_MCTS`

## Authoritative Game Source

`C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes\src`

## Goal

- Compare every unit, rule, action, achievement, and turn transition in the Java game source with native MCTS, one mechanic at a time.
- Treat the Java engine as authoritative. Native code should simulate the information the bot truly has, including fog-of-war limits and random outcomes, as closely as possible.

## Audit Loop

- For each mismatch, update the parity test suite so the failure is covered automatically next time.
- Spawn a narrow gpt-5.5 low subagent whose only job is fixing that specific mismatch.
- Wait for the subagent to finish, then verify the patch yourself. If the failure remains, keep working the same mechanic until it is resolved, then move to the next one.

## Operating Constraints

- Do not let subagents drift into the full project. Their scope is the mismatch you hand them, not the audit as a whole.
- Do not check their patch constantly. Wait roughly three minutes between checks when they are running.
- Commit only after the fix is verified by tests.
- Keep using the external Java repo at `C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes` as the source of truth for rules and transitions.

## Current Verified State

- The hidden enemy end-turn and hidden enemy attack parity issues are resolved in native rules.
- The latest verified commit is `59bd12e`: Fix native hidden action parity.
- The full `py/tests/test_native_mcts.py` suite passes at the time of this reference.

## Resume Point

- If context is lost, start by checking `git status`, then resume from the latest unresolved parity failure in `py/search/native/native_rules.cpp` or the parity runner tests.
- If a new mismatch appears, record it in tests before doing anything else.

## Note

This reference should be updated as the parity audit advances.

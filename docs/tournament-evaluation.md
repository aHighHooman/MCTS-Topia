# Tournament Evaluation

This is the repository's procedure for comparing the primitive and `turn-macro-exp` static bots.

## Objective handling

Use the Java `Tournament` runner with `Turn Limit: 0` for strength comparisons.

For MIGHT, this leaves the normal capital-control objective enabled. A completed match therefore has a real winner and loser. The tournament summary should report every game with `winner=...`, and the final standings should show no failed matches or ties.

Do not use a positive `Turn Limit` as a neutral observation cutoff. In the Java rules, reaching a finite turn limit invokes ranking: the top-ranked tribe is assigned `WIN` and the others `LOSS`. Ranking uses score, researched technologies, cities, and production. Those results are adjudicated outcomes, not evidence that one bot achieved the normal game objective.

## Required tournament shape

Use balanced seats and mirrored seeds so seat order does not decide the comparison:

```json
{
  "Game Mode": "Might",
  "Map Type": "Drylands",
  "Map Size": "Tiny",
  "Turn Limit": 0,
  "Parallel Games": 8,
  "Balance Seats": true,
  "Match Retry Limit": 0,
  "External Action Timeout Ms": 120000
}
```

Use deterministic bot seeds, keep the primitive and macro commands identical except for their search mode and calibrated search settings, and write each experiment to a unique directory under `debug-logs/`.

Run the Java tournament with the external game's JSON jar:

```powershell
& "$env:JAVA_HOME\bin\java.exe" `
  -cp "out;C:\Users\Umair\OneDrive\Desktop\Work\Self_Projects\TribesTopia\Tribes\lib\json.jar" `
  Tournament debug-logs\<experiment>\tournament.json
```

Afterward inspect the summary and stderr logs:

```powershell
Get-Content debug-logs\<experiment>\summary.log -Tail 120
Get-ChildItem debug-logs\<experiment>\external_logs -Recurse -Filter *.stderr.log |
  Where-Object Length -gt 0
```

## Equalizing search budgets

Do not equalize the raw simulation counts. A primitive simulation advances one primitive-action tree path, while a macro outer simulation can run an inner primitive search and materialize a complete turn plan. Their counts are different units of work.

Instead:

1. Run short wall-clock probes for each bot on the same payload and hardware.
2. Pick fixed per-mode simulation budgets that produce comparable time per action.
3. Use those fixed budgets in the tournament so every match has deterministic, reproducible work.
4. Record the probe target, payload, budgets, and resulting elapsed times in `docs/experiment-record.md`.

The current normal-objective series used approximately `0.15` seconds per action: primitive `4500` simulations and macro `46` outer simulations with inner `128` simulations. These are calibration values, not a claim that the raw counts are equivalent.

For wall-clock behavior checks, use `--wall-clock-per-action-seconds` directly on both bot commands. Do not mix wall-clock and fixed-simulation modes in one strength claim.

## Interpreting results

Count only games that completed the normal objective. Before comparing win rates, verify:

- `Turn Limit: disabled` appears in the tournament header;
- every game has a winner and a normal completion turn;
- failed matches are zero;
- external stderr logs are empty or understood;
- both seat assignments were played.

Keep short smoke runs for crash and protocol detection. Use longer held-out seed ranges for strength claims. A neutral or non-regressive smoke result is not a general improvement.

The detailed history and rejected hypotheses are recorded in [experiment-record.md](experiment-record.md).

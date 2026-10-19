# Eval results: single agent vs. multi-agent

| Questions | n | Coverage (sections, multi ÷ single) | Tokens (multi ÷ single) | Parallel speed-up | Completed (single / multi) | Tool errors | Recovered |
|---|---|---|---|---|---|---|---|
| all | 8 | 2.4× | 2.7× | 2.5× | 100% / 100% | 2 | 50% |
| broad | 5 | 2.9× | 2.8× | 2.7× | 100% / 100% | 2 | 50% |
| narrow | 3 | 1.7× | 2.5× | 2.3× | 100% / 100% | 0 | – |

## Answer quality (LLM judge: google/gemini-2.5-flash)

| Questions | Judged | Completeness (multi / tie / single) | Specificity (multi / tie / single) | Faithfulness (single / multi) |
|---|---|---|---|---|
| all | 8 | 7 / 1 / 0 | 8 / 0 / 0 | 99% / 95% |
| broad | 5 | 5 / 0 / 0 | 5 / 0 / 0 | 100% / 97% |
| narrow | 3 | 2 / 1 / 0 | 3 / 0 / 0 | 98% / 92% |

## Per question

| ID | Breadth | Sections | Papers | Tokens | Seconds | Speed-up | More complete | More specific | Faithfulness (single / multi) |
|---|---|---|---|---|---|---|---|---|---|
| q1 | broad | 2 → 9 | 2 → 6 | 97,623 → 190,797 | 43 → 99 | 2.8× | multi | multi | 14/14 / 50/53 |
| q2 | broad | 3 → 7 | 1 → 3 | 105,103 → 197,215 | 48 → 80 | 2.6× | multi | multi | 23/23 / 69/69 |
| q3 | broad | 4 → 9 | 2 → 5 | 100,810 → 252,980 | 45 → 79 | 2.5× | multi | multi | 31/31 / 59/61 |
| q4 | broad | 4 → 10 | 2 → 5 | 68,053 → 274,425 | 36 → 75 | 2.7× | multi | multi | 23/23 / 50/53 |
| q5 | broad | 4 → 11 | 2 → 6 | 79,978 → 287,823 | 45 → 80 | 2.8× | multi | multi | 33/33 / 53/53 |
| q6 | narrow | 5 → 11 | 2 → 5 | 80,219 → 231,664 | 41 → 69 | 2.7× | multi | multi | 20/21 / 51/53 |
| q7 | narrow | 3 → 4 | 1 → 1 | 50,271 → 112,682 | 37 → 51 | 1.9× | multi | multi | 27/27 / 33/33 |
| q8 | narrow | 5 → 8 | 1 → 1 | 46,373 → 111,813 | 42 → 54 | 2.3× | tie | multi | 39/40 / 53/63 |

Tokens count the research model's calls; extraction and the supervisor's brief and answer calls are not included.
Parallel speed-up = total worker time ÷ slowest worker (the research phase only).
Judge winners count only when both answer orders agree; disagreements are ties. Faithfulness = claims supported by their cited findings ÷ all factual claims.
Judge calls: 32 · 96,953 tokens. Every metric, verdict, and judge call for a question is saved in traces/eval/<id>_scores.json.

# Bounded temporal candidate-filter experiment

Effective reads previously constructed a Python fact object for every historical
row in the requested scope. SQLite now selects conservative candidates first:
one second on either side of each bound, retaining bounds it cannot parse.
Python still performs the exact half-open interval comparison. There is no
schema migration or new index. Unparseable SQLite timestamps include valid
Python ISO forms, such as offsets containing seconds and basic date notation.

The offline synthetic comparison used 100, 1,000 and 10,000 facts in the queried
scope, ten keys, and equal-sized unrelated-owner and unrelated-context scopes.
Both arms returned the same ten expected fact IDs on every measured read.

| Facts in scope | Original median across repeats | Candidate median across repeats | Original traced peak | Candidate traced peak |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.534–0.591 ms | 0.118–0.132 ms | 207 KB | 22 KB |
| 1,000 | 5.814–5.973 ms | 0.445–0.489 ms | 2.13 MB | 22 KB |
| 10,000 | 67.762–73.982 ms | 7.133–8.327 ms | 22.61 MB | 22 KB |

Each of three connection-reopen repeats used five warmups and thirty measured
reads per arm, alternating arm order. Allocation measurement ran separately.
Figures measure direct temporal reads on one Windows host. Traced Python heap
peaks are not total process RSS or SQLite allocation measurements. OS caches
were not flushed. These results do not measure model calls or complete recall.
The SQL scan still scales with history size; the gain comes from avoiding most
Python object construction. Workloads with many active keys or unparseable
timestamps may retain substantially more candidates.

The provenance-verified run lives in the task artifact `temporal-resource-v2`.
It loaded only frozen source copies and checked their SHA256 before and after
the run, along with every synthetic database. Baseline source SHA256:
`876542c36b5b6797620e7a1afd68bbb71248820c08ae1b408f5d0b94e9267385`.
Candidate source SHA256:
`163ca04799967db5622baac87026a7c424c1a6f1fe8df7fae824673f52b8e158`.
Earlier V1 evidence is preserved but marked preliminary because its copied
baseline differed from the recorded hash and no final source check was made.

To reproduce from a checkout with development dependencies, save the desired
baseline temporal store source and run:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
python scripts/benchmark_temporal_reads.py --baseline-file ../baseline_temporal.py --candidate-file morgan_brain/memory/store/temporal.py --output-dir ../fresh-temporal-benchmark
```

The output directory must be new. The script accepts source files only, creates
its own synthetic databases, and opens them read-only during comparison. It
records source snapshots, database hashes, every timing sample and environment
metadata in that directory. No model or network request is made.

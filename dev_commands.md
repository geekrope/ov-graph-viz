# Dev commands

All commands are run from the project root (`intel task/`), not from `src/`.

## Tests

```
pytest -q                          # all tests
pytest tests/test_graph.py -v      # one file
pytest -k shared                   # tests whose name contains "shared"
```

## Coverage

Requires `pip install pytest-cov`.

```
pytest --cov=src --cov-report=term-missing               # table + missing line numbers
pytest --cov=src --cov-branch --cov-report=term-missing  # also count if/else branches
pytest --cov=src --cov-report=html                       # HTML report
start htmlcov/index.html                                 # open it (uncovered lines are red)
```

## Graphviz profiling

1. Save the dot source from Python (e.g. in `main.py`):

   ```python
   dg = GraphvizRenderer().build(graph)
   dg.save("../graph.dot")
   ```

2. Run dot with verbose phase timings. Use `cmd /c` - PowerShell `2>` writes
   the log in UTF-16 and prepends a NativeCommandError:

   ```
   cmd /c "dot -v -Tsvg graph.dot -o graph.svg 2> dot_log.txt"
   ```

   ```
   dot -v -Tsvg simple_graph.dot -o simple_graph.svg 2> dot_log.txt
   ```

3. Extract the phase timings:

   ```
   findstr /C:"Starting phase" /C:"Maxrank" /C:"crossings" /C:"routesplines" /C:"gvRenderJobs" dot_log.txt
   findstr /R /C:"network simplex: *[0-9][0-9][0-9]* nodes.*sec" dot_log.txt
   ```

   | log line                                   | phase                          |
   |--------------------------------------------|--------------------------------|
   | `Starting phase 1 [dot_rank]`              | ranking (levels)               |
   | `mincross %N: X crossings, T secs`         | crossing minimization          |
   | `Starting phase 3 [dot_position]` + `network simplex: K nodes ... T sec` | coordinates; K = real + virtual nodes |
   | `routesplines: E edges ... T sec`          | edge routing                   |
   | `gvRenderJobs: T secs`                     | SVG output                     |

4. Try layout options without changing code (`-G` graph, `-N` node, `-E` edge attrs):

   ```
   cmd /c "dot -v -Gsplines=polyline -Tsvg graph.dot -o out.svg 2> log_polyline.txt"
   cmd /c "dot -v -Gmclimit=0.1 -Gremincross=false -Tsvg graph.dot -o out.svg 2> log_mincross.txt"
   cmd /c "dot -v -Gnslimit=2 -Gnslimit1=2 -Tsvg graph.dot -o out.svg 2> log_nslimit.txt"
   ```

5. Graphviz version for the README environment section: `dot -V`

Baseline (2026-10-12, TinyLlama int8, ConstantHiding + Truncation(1200)):
total ~1.1 s; position 0.94 s (6705 nodes incl. virtual vs 1200 real),
mincross 0.02 s, splines ~0.05 s, SVG 0.05 s.

Same model, Truncation(1500): 1500 nodes / 1569 edges, Maxrank 546,
38835 nodes in position phase (real + virtual).

| options                       | position | splines        | total   |
|-------------------------------|----------|----------------|---------|
| default                       | 76.15 s  | 0.20 s         | ~77 s   |
| -Gnslimit=2                   | 3.80 s   | 170.86 s       | ~175 s  |
| -Gnslimit=2 -Gsplines=polyline| 3.98 s   | > 150 s (killed)| > 150 s|
| -Gsplines=line                | 96.69 s  | 0.03 s         | ~98 s   |
| -Gnslimit=2 -Gsplines=line    | 2.77 s   | 0.03 s         | ~4 s    |

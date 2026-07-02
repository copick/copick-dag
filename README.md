# copick-dag

Define [copick](https://github.com/copick/copick) data-processing pipelines **once** as an
abstract DAG, then materialize them to different execution backends.

Today, copick pipelines are shell scripts that call the `copick` CLI. That is reproducible but:

- each command runs across the **whole** run set before the next starts (no parallelism *along*
  the pipeline, only *within* a command);
- nothing persists in memory between commands — every step re-parses the project and re-reads
  zarr;
- the dependency structure (a DAG) is implicit in the input/output URIs and never made explicit.

`copick-dag` makes that DAG explicit and backend-agnostic:

```
GraphSpec (YAML/JSON, LLM-authorable)
   │  planner (opens the copick project once)
   ▼
ResolvedPlan  (concrete DAG: one task per (node, run, voxel-spacing))
   │
   ├── bash      →  a `copick …` CLI script (same behaviour as today)
   ├── memory    →  in-process, write-through memory overlay, parallel across runs + along graph
   └── nextflow  →  (roadmap) per-run process parallelism for HPC
```

## Status

Early alpha. Implemented:

- **IR + registry + planner** — declarative graph, operation registry (introspects the installed
  `copick` CLI + a self-contained overlay), run + voxel-spacing axes, URI edge inference.
- **bash** emitter and **in-memory** write-through executor — voxel-identical on `example_1`.
- **Dynamic frontier** — `sweep`/CSV/`switch`/`=`-expr expansion, `split` fan-out, `*`-glob fan-in,
  `coalesce`/content routing, and a cross-run `seg-stats` barrier; reproduces `example_3`
  voxel-identical on both backends.
- **nextflow** emitter (emit-only) — one process per task, done-signal channels, `local`+`slurm`
  profiles; `copick-dag render/run --backend nextflow`.
- **MCP tools** — `copick_dag.api` (JSON-safe describe-ops / schema / validate / plan / render /
  visualize / run) is exposed as graph tools on the **copick-mcp** server, so an LLM can author,
  inspect, and run pipelines end-to-end.

Note: the in-memory backend defaults to sequential-per-run (`workers=1`) because copick's
`CopickRoot` is not thread-safe to mutate; `>1` isolates a root per run.

## Key ideas

- **Edges are URIs.** A node that writes `membrane:temp/x@20` and one that reads it are
  connected automatically; explicit `needs:`/`after:` overrides resolve ambiguity.
- **Voxel spacing is a sweep axis**, not a literal — discover it from the project (copick matches
  spacings by exact float, so `10.0` never matches a stored `10.006`; the planner adds tolerance).
- **Reuse copick's per-run workers.** The in-memory backend swaps the *storage* underneath the
  existing operation functions via a write-through overlay, so array algorithms are not
  reimplemented.

## Install (development)

```bash
pip install -e .
```

Requires an environment that already has `copick` and `copick-utils` installed.

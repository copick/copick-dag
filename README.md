# copick-dag

Define [copick](https://github.com/copick/copick) data-processing pipelines **once** as an
abstract DAG, then materialize them to different execution backends.

Today, copick pipelines are shell scripts that call the `copick` CLI. That is reproducible but:

- each command runs across the **whole** run set before the next starts (no parallelism *along*
  the pipeline, only *within* a command);
- nothing persists in memory between commands — every step re-parses the project and re-reads
  Zarr;
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
   └── nextflow  →  per-run process parallelism for local and SLURM execution
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

## Installation

copick-dag 2.0 requires Python 3.11 or newer and the copick 2.0 ecosystem. The lean installation includes core and
copick-utils operations without installing PyTorch:

```bash
pip install --pre copick-dag
```

Four registered operations are supplied by copick-torch: `convert.seg2slab`, `convert.picks2slab`,
`process.downsample`, and `process.bandpass`. Install the explicit extra when those operations are needed:

```bash
pip install --pre "copick-dag[torch]"
```

Operation descriptions report whether their providing command is available. Selecting a missing optional operation
produces an installation hint; ordinary planning and execution remain torch-free.

## OME-Zarr and copy/move behavior

The 2.0 line consumes the OME-Zarr 0.5 / Zarr v3 contracts provided by `copick>=2.0.0-alpha.1` and
`copick-utils>=2.0.0-alpha.1`. It can process supported legacy OME-Zarr 0.4 / Zarr v2 projects and supported
noncanonical Zarr v3 layouts. New numerical outputs are written by upstream copick APIs in their canonical
OME-Zarr 0.5 / Zarr v3 representation; copick-dag does not implement its own Zarr reader or writer.

`cp` and `mv` guarantee decoded entity equivalence, identity metadata, and the requested overwrite behavior. The
bash and Nextflow backends may preserve a source store byte-for-byte, while the memory backend may decode and
rewrite it using the current canonical encoding. Raw keys, chunk/shard layout, codecs, and source format are not
part of the cross-backend copy contract.

Nextflow `-resume` is intended for crash recovery. Changes made directly to an external copick store do not alter
Nextflow's task hashes, so a fresh run is required after such changes.

## Development

```bash
uv sync --locked --extra test --extra dev
uv run pytest -m "not torch_plugin"
```

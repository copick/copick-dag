"""M5: the JSON-safe api layer that MCP tools call."""

import json
import os

import yaml

import copick_dag.api as api
from fixtures import M3A_RUNS, M3A_VS, make_m3a_csv, make_m3a_fixture, read_segmentations

PERRUN = os.path.join(os.path.dirname(__file__), "data", "example_3_perrun.yaml")


def _graph_str(csv, config="/abs/config.json", sort_keys=False):
    g = yaml.safe_load(open(PERRUN))
    g["project"]["config"] = config
    g["sweep"]["over"][0]["csv"] = csv
    g["sweep"]["over"][0]["runs"] = M3A_RUNS
    return yaml.safe_dump(g, sort_keys=sort_keys)


def test_describe_ops_and_op(tmp_path):
    from copick_dag.registry import all_ops

    ops = api.describe_ops()
    assert len(ops["ops"]) == len(all_ops()) and len(ops["ops"]) >= 41  # includes the M6 transform ops
    assert all(o.get("help") for o in ops["ops"] if o["available"])
    torch_ops = [o for o in ops["ops"] if o.get("plugin") == "copick-torch"]
    assert len(torch_ops) == 4 and all(o["install_extra"] == "torch" for o in torch_ops)
    segop = api.describe_op("logical.segop")
    assert segop["op_id"] == "logical.segop" and segop["scope"] == "per_run"
    assert any(s["slot"] == "i" for s in segop["inputs"])
    stats = api.describe_op("process.seg-stats")
    assert stats["scope"] == "barrier"
    json.dumps(ops)  # serializable


def test_validate_selected_missing_plugin_is_actionable():
    from copick_dag.registry import get_op
    from copick_dag.registry.introspect import operation_available

    if operation_available(get_op("process.downsample")):
        return
    graph = {
        "version": 1,
        "project": {"config": "unused.json"},
        "runs": ["run1"],
        "nodes": [
            {
                "id": "downsample",
                "op": "process.downsample",
                "in": {"i": "wbp@10.0"},
                "out": {"o": "wbp@20.0"},
            },
        ],
    }
    result = api.validate_graph(graph)
    assert not result["valid"]
    assert "copick-dag[torch]" in " ".join(result["errors"])


def test_graph_schema_has_models():
    sch = api.graph_schema()
    defs = sch.get("$defs", {})
    assert any("GraphNode" in k for k in defs) and any("Sweep" in k for k in defs)
    json.dumps(sch)


def test_load_graph_from_dict_and_str(tmp_path):
    csv = make_m3a_csv(str(tmp_path / "c.csv"))
    s = _graph_str(csv)
    from_str = api.load_graph(s)
    from_dict = api.load_graph(yaml.safe_load(s))
    assert from_str.to_dict() == from_dict.to_dict()


def test_validate_and_plan(tmp_path):
    csv = make_m3a_csv(str(tmp_path / "c.csv"))
    g = _graph_str(csv)
    v = api.validate_graph(g)
    assert v["valid"] and v["task_count"] == 28 and sorted(v["runs"]) == sorted(M3A_RUNS)
    p = api.plan_graph(g)
    # final2 (combine over *:expsplit) depends on E (cyto_exp) and F (split_expand)
    by_id = {t["task_id"]: t for t in p["tasks"]}
    final2 = next(t for t in p["tasks"] if t["node_id"] == "final2__L1" and t["run"] == "run_mt")
    dep_nodes = {by_id[d]["node_id"] for d in final2["deps"]}
    assert {"cyto_exp__L1", "split_expand__L1"} <= dep_nodes
    json.dumps(p)


def test_validate_reports_errors_not_raises(tmp_path):
    v = api.validate_graph("nodes: [{op: does.not.exist, in: {}, out: {}}]\nproject: {config: x}")
    assert v["valid"] is False and v["errors"]


def test_bind_order_independent(tmp_path):
    # An LLM may write `bind` keys in any order; the CSV id-namespacing must not depend on it.
    csv = make_m3a_csv(str(tmp_path / "c.csv"))
    g = yaml.safe_load(_graph_str(csv))
    filt = next(n for n in g["nodes"] if n.get("id") == "filt")
    filt["expand"]["bind"] = {"min_vox": 2, "name": 1}  # reversed order
    v = api.validate_graph(yaml.safe_dump(g))
    assert v["valid"] and v["task_count"] == 28


def test_render_bash_and_nextflow(tmp_path):
    csv = make_m3a_csv(str(tmp_path / "c.csv"))
    g = _graph_str(csv)
    b = api.render_graph(g, backend="bash")
    assert b["backend"] == "bash" and "copick process split" in b["script"]
    n = api.render_graph(g, backend="nextflow")
    # One process per node (not per run x node): the M3A graph has 14 nodes over 2 runs (28 tasks),
    # and the runs fan out over a channel -- so the module is per-node and run-count-independent.
    assert n["script"].count("process p_") == 14 and "runsList" in n["script"]
    assert "slurm" in n["nextflow_config"]


def test_visualize(tmp_path):
    csv = make_m3a_csv(str(tmp_path / "c.csv"))
    viz = api.visualize_graph(_graph_str(csv))
    assert viz["mermaid"].startswith("graph TD")
    assert "-->" in viz["mermaid"] and viz["text"].startswith("# ")


def test_run_graph_memory(tmp_path):
    csv = make_m3a_csv(str(tmp_path / "c.csv"))
    cfg = make_m3a_fixture(str(tmp_path / "proj"))
    g = _graph_str(csv, config=cfg)
    res = api.run_graph(g, config=cfg, backend="memory")
    assert res["ok"] and not [e for e in res["errors"] if e.startswith("[ERROR]")]
    finals = read_segmentations(cfg, M3A_RUNS, "final2", "final", M3A_VS)
    assert len(finals) == 2  # multilabel:final2/final produced for both runs

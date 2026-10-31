from collections import Counter

from skillgraph.graph.schema import Label, Rel


def test_relationship_counts_match_the_source(ctx):
    counts = Counter(d["type"] for *_, d in ctx.store.g.edges(data=True))
    assert counts[Rel.REQUIRED_BY.value] == 74
    assert counts[Rel.PREREQUISITE_OF.value] == 30
    assert counts[Rel.HAS.value] == 12
    # embedded in node data rather than the edge lists
    assert counts[Rel.CONTAINS.value] == sum(len(s.track_ids) for s in ctx.dataset.skills)
    assert counts[Rel.TEACHES.value] == sum(len(c.skill_ids) for c in ctx.dataset.courses)


def test_relationship_directions_follow_the_brief(ctx):
    g = ctx.store.g
    assert g.has_edge("SEC-01", "TRK-01", Rel.HAS.value)                 # Sector HAS Track
    assert g.has_edge("TRK-01", "SKL-04", Rel.CONTAINS.value)            # Track CONTAINS Skill
    assert g.has_edge("SKL-04", "ROL-01", Rel.REQUIRED_BY.value)         # Skill REQUIRED_BY Role
    assert g.has_edge("SKL-01", "SKL-04", Rel.PREREQUISITE_OF.value)     # Skill PREREQUISITE_OF Skill
    assert g.has_edge("CRS-02", "SKL-04", Rel.TEACHES.value)             # Course TEACHES Skill


def test_economy_nodes_link_every_sector(ctx):
    for sector in ctx.store.ids(Label.SECTOR):
        targets = [v for _, v, d in ctx.store.g.out_edges(sector, data=True) if d["type"] == Rel.IN_ECONOMY.value]
        assert len(targets) == 1 and ctx.store.has(targets[0], Label.ECONOMY)


def test_typed_lookups(ctx):
    s = ctx.store
    assert s.required_skills("ROL-01") == ["SKL-01", "SKL-02", "SKL-03", "SKL-04", "SKL-51", "SKL-52"]
    assert "ROL-01" in s.roles_requiring("SKL-04")
    assert s.prerequisites("SKL-04") == ["SKL-01", "SKL-52"]
    assert s.courses_teaching("SKL-04") == ["CRS-02"]
    assert s.prerequisite_strength("SKL-08", "SKL-09").value == "soft"


def test_export_is_complete(ctx):
    out = ctx.store.export()
    assert len(out["nodes"]) == ctx.store.g.number_of_nodes()
    assert len(out["edges"]) == ctx.store.g.number_of_edges()


def test_cypher_export_covers_every_node_and_relationship(ctx):
    from skillgraph.graph.neo4j_export import to_cypher

    script = to_cypher(ctx.store)
    assert script.count("MERGE (n:") == ctx.store.g.number_of_nodes()
    assert script.count("MERGE (a)-[r:") == ctx.store.g.number_of_edges()
    assert "MATCH (a:Skill {id: \"SKL-01\"}), (b:Skill {id: \"SKL-04\"}) MERGE (a)-[r:PREREQUISITE_OF]->(b)" in script

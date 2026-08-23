from src.nlp.dataset_generator import INCIDENT_TYPES, SEVERITY_LEVELS, generate_dataset


def test_generates_at_least_100_incidents():
    records = generate_dataset(n_per_type=12, seed=1)
    assert len(records) >= 100


def test_covers_all_incident_types():
    records = generate_dataset(n_per_type=5, seed=2)
    types_seen = {r["incident_type"] for r in records}
    assert types_seen == set(INCIDENT_TYPES.keys())


def test_severity_values_are_valid():
    records = generate_dataset(n_per_type=5, seed=3)
    assert all(r["severity"] in SEVERITY_LEVELS for r in records)


def test_each_incident_has_relevant_articles():
    records = generate_dataset(n_per_type=5, seed=4)
    assert all(len(r["relevant_articles"]) > 0 for r in records)


def test_jargon_present_in_dataset():
    records = generate_dataset(n_per_type=12, seed=5)
    all_text = " ".join(r["narrative_text"] for r in records)
    for term in ["CAEX", "chancador", "acuñadura", "faena", "EPP"]:
        assert term.lower() in all_text.lower(), f"Termino '{term}' no aparece en el dataset generado"


def test_deterministic_with_seed():
    a = generate_dataset(n_per_type=5, seed=42)
    b = generate_dataset(n_per_type=5, seed=42)
    assert [r["narrative_text"] for r in a] == [r["narrative_text"] for r in b]

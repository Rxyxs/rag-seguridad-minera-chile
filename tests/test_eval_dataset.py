"""Conjunto de evaluacion sintetico de operaciones mineras: esquema, balance, determinismo y respuestas recalculadas."""
import json
import re
from datetime import date
from random import Random

import pytest

from src.evaluation.dataset import generator as G

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
KEYS = {"question", "contexts", "ground_truth", "domain"}


def numbers(text: str) -> list[float]:
    """Numeros de un texto en formato chileno (``1.150,5``), sin identificadores de equipo, turnos, lineas, horas ni P80."""
    text = re.sub(r"\d+ de [a-z]+ de \d{4}|[A-Z]{3,5}-\d+|Turno [A-D]|[Ll]ínea \d|\d{2}:\d{2}|P80", " ", text)
    return [float(t.replace(".", "").replace(",", ".")) for t in re.findall(r"\d[\d.]*(?:,\d+)?", text)]


def has(truth: str, value: float, tol: float = 0.051) -> bool:
    return any(abs(v - value) <= tol for v in numbers(truth))


def parse_dates(text: str) -> list[date]:
    return [date(int(y), MESES.index(m) + 1, int(d)) for d, m, y in re.findall(r"(\d+) de (\w+) de (\d{4})", text)]


# ------------------------------------------------------------------------ formatos y formulas
def test_chilean_number_format():
    assert G.es(1234.5, 1) == "1.234,5" and G.es(9200) == "9.200" and G.es(0.62, 2) == "0,62" and G.es(1_000_000) == "1.000.000"


def test_plural_helper():
    assert G.plural(1, "accidente", "accidentes") == "1 accidente" and G.plural(0, "accidente", "accidentes") == "0 accidentes"


def test_formulas_by_hand():
    assert G.throughput(9600, 12) == 800
    assert G.specific_energy_kwh_t(120, 10_000) == 12.0
    assert G.pct_change(6000, 6600) == pytest.approx(10.0) and G.pct_change(8000, 7600) == pytest.approx(-5.0)
    assert G.pct_over(165, 150) == pytest.approx(10.0)
    assert G.circulating_load_pct(2500, 1000) == 250.0
    assert G.recovery_pct(10_000, 0.5, 40.0) == pytest.approx(80.0)          # 50 t contenidas, 40 t recuperadas
    assert G.mtbf(2000, 5) == 400
    assert G.availability_pct(720, 72) == pytest.approx(90.0)
    assert G.frequency_index(2, 500_000) == pytest.approx(4.0)               # 2 por 0,5 millones de HH


# ----------------------------------------------------------------------------------- esquema
@pytest.fixture(scope="module")
def data():
    return G.generate()


def test_there_are_at_least_30_pairs_and_the_domains_are_balanced(data):
    assert len(data) >= 30
    counts = {d: sum(i["domain"] == d for i in data) for d in G.DOMAINS}
    assert set(counts) == {"mantenimiento", "proceso", "seguridad"} and min(counts.values()) >= 10


def test_every_item_follows_the_schema_exactly(data):
    for item in data:
        assert set(item) == KEYS
        assert isinstance(item["question"], str) and item["question"].strip().endswith("?")
        assert isinstance(item["ground_truth"], str) and item["ground_truth"].strip()
        assert item["domain"] in G.DOMAINS
        assert isinstance(item["contexts"], list) and 1 <= len(item["contexts"]) <= 3
        assert all(isinstance(c, str) and c.strip() for c in item["contexts"])


def test_questions_are_unique(data):
    qs = [i["question"] for i in data]
    assert len(qs) == len(set(qs))


def test_some_items_carry_a_distractor_context(data):
    assert 10 <= sum(len(i["contexts"]) > 1 for i in data) <= len(data) - 10


def test_generation_is_deterministic_and_the_seed_matters():
    assert G.generate(seed=3) == G.generate(seed=3)
    assert G.generate(seed=3) != G.generate(seed=4)


def test_a_larger_set_is_possible_and_still_unique():
    big = G.generate(per_family=3)
    assert len(big) == 54 and len({i["question"] for i in big}) == 54


def test_export_writes_strict_utf8_json(tmp_path):
    path = tmp_path / "qa.json"
    items = G.export(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == items and "¿" in path.read_text(encoding="utf-8")  # sin escapes \u: legible y en UTF-8


# --------------------------------------------------- cada respuesta se recalcula desde su contexto
def first(family, seed=1, distract=False):
    item = family(Random(seed), distract)
    return item, numbers(item["contexts"][0]), item["ground_truth"]


@pytest.mark.parametrize("seed", [1, 2, 3])
class TestAnswersAgreeWithTheirContext:
    def test_throughput(self, seed):
        _, (tons, hours), truth = first(G.p_throughput, seed)
        assert has(truth, tons / hours)

    def test_specific_energy(self, seed):
        _, (mwh, tons), truth = first(G.p_energy, seed)
        assert has(truth, mwh * 1000 / tons)

    def test_power_change(self, seed):
        _, (before, after), truth = first(G.p_power_change, seed)
        assert has(truth, abs(after - before) / before * 100) and ("aumentó" in truth) == (after > before)

    def test_p80(self, seed):
        _, (measured, target), truth = first(G.p_p80, seed)
        if measured == target:
            assert "en la meta" in truth
        else:
            assert has(truth, abs(measured - target)) and has(truth, abs(measured - target) / target * 100)
            assert ("sobre la meta" in truth) == (measured > target)

    def test_circulating_load(self, seed):
        _, (fresh, under), truth = first(G.p_circulating, seed)
        assert has(truth, under / fresh * 100, 0.51)

    def test_recovery(self, seed):
        item, (feed, grade, fine), truth = first(G.p_recovery, seed)
        assert has(truth, fine / (feed * grade / 100) * 100)

    def test_mtbf(self, seed):
        _, (hours, fails), truth = first(G.m_mtbf, seed)
        assert has(truth, hours / fails)

    def test_availability(self, seed):
        _, (calendar, down), truth = first(G.m_availability, seed)
        assert calendar == 720 and has(truth, (calendar - down) / calendar * 100)

    def test_next_pm(self, seed):
        _, (now, last, interval), truth = first(G.m_next_pm, seed)
        assert has(truth, last + interval, 0.5) and has(truth, last + interval - now, 0.5) and now < last + interval

    def test_temperature(self, seed):
        _, (reading, limit), truth = first(G.m_temperature, seed)
        assert reading > limit and has(truth, reading - limit, 0.5)

    def test_backlog(self, seed):
        _, (a, b), truth = first(G.m_backlog, seed)
        assert has(truth, a + b, 0.5)

    def test_remaining_life(self, seed):
        _, (life, used), truth = first(G.m_remaining_life, seed)
        assert used < life and has(truth, life - used, 0.5) and has(truth, used / life * 100)

    def test_frequency_index(self, seed):
        _, (accidents, man_hours), truth = first(G.s_frequency, seed)
        assert has(truth, accidents * 1_000_000 / man_hours, 0.006)

    def test_ramp_speed_limit(self, seed):
        item, (slope, loaded, empty), truth = first(G.s_ramp_limit, seed)
        assert f"{int(slope)}%" in item["question"] and has(truth, loaded, 0.5) and not has(truth, empty, 0.5)

    def test_co_excess(self, seed):
        _, (reading, limit), truth = first(G.s_co, seed)
        assert reading > limit and has(truth, reading - limit, 0.5) and "evacuar" in truth

    def test_shift_events(self, seed):
        _, (a, b, c), truth = first(G.s_shift_events, seed)
        assert has(truth, a + b + c, 0.5)

    def test_min_distance(self, seed):
        _, (metres,), truth = first(G.s_min_distance, seed)
        assert has(truth, metres, 0.5)

    def test_days_without_accident(self, seed):
        item, _, truth = first(G.s_days_without, seed)
        last, today = parse_dates(item["contexts"][0])
        assert today > last and has(truth, (today - last).days, 0.5)

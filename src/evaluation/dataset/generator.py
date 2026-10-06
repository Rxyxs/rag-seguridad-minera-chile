"""Conjunto de evaluacion SINTETICO de preguntas y respuestas sobre operaciones mineras, con verdad conocida.

    python -m src.evaluation.dataset.generator          # escribe tmp_agent_a/ground_truth_qa.json

Cada par se construye a partir de parametros numericos sorteados con semilla, y la respuesta se calcula con una formula
(no se escribe a mano), asi que la respuesta es verificable contra el contexto. Sirve para medir si un sistema de
recuperacion y respuesta (RAG) extrae y combina bien datos de un reporte; **no son datos de ninguna faena real**: los
equipos, turnos, lecturas y limites son inventados, y los limites de seguridad son ilustrativos, no normativa.

Esquema de cada elemento: ``question``, ``contexts`` (lista de textos; a veces incluye uno distractor de otro equipo),
``ground_truth`` y ``domain`` (``mantenimiento``, ``proceso`` o ``seguridad``).

Familias por dominio (2 instancias de cada una, 36 pares con la configuracion por defecto):

* proceso (telemetria de molienda): throughput, energia especifica, variacion de potencia, desviacion del P80,
  carga circulante, recuperacion de cobre.
* mantenimiento (camiones CAEX): MTBF, disponibilidad, horas al proximo PM, exceso de temperatura, backlog de OTs,
  vida util restante de un componente.
* seguridad (reporte de turno): indice de frecuencia, limite de velocidad en rampa (lectura), exceso de CO,
  eventos del turno, distancia minima de aproximacion (lectura), dias sin accidente con tiempo perdido.
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "tmp_agent_a" / "ground_truth_qa.json"
DOMAINS = ("mantenimiento", "proceso", "seguridad")
SEED = 7
PER_FAMILY = 2


# ------------------------------------------------------------------------------ formatos
def es(x: float, d: int = 0) -> str:
    """Numero con separador de miles ``.`` y decimal ``,`` (convencion chilena)."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def plural(count: int, one: str, many: str) -> str:
    """'1 accidente' / '4 accidentes'."""
    return f"{count} {one if count == 1 else many}"


def es_date(d: date) -> str:
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    return f"{d.day} de {meses[d.month - 1]} de {d.year}"


# ------------------------------------------------------------------------------ formulas
def throughput(tons: float, hours: float) -> float:
    return tons / hours


def specific_energy_kwh_t(energy_mwh: float, tons: float) -> float:
    return energy_mwh * 1000.0 / tons


def pct_change(before: float, after: float) -> float:
    return (after - before) / before * 100.0


def pct_over(measured: float, target: float) -> float:
    return (measured - target) / target * 100.0


def circulating_load_pct(underflow_tph: float, fresh_feed_tph: float) -> float:
    return underflow_tph / fresh_feed_tph * 100.0


def recovery_pct(feed_t: float, grade_pct: float, fine_cu_t: float) -> float:
    return fine_cu_t / (feed_t * grade_pct / 100.0) * 100.0


def mtbf(hours: float, failures: int) -> float:
    return hours / failures


def availability_pct(calendar_hours: float, downtime_hours: float) -> float:
    return (calendar_hours - downtime_hours) / calendar_hours * 100.0


def frequency_index(accidents: int, man_hours: float) -> float:
    """Indice de frecuencia: accidentes con tiempo perdido por millon de horas hombre trabajadas."""
    return accidents * 1_000_000.0 / man_hours


# --------------------------------------------------------------------- piezas reutilizables
def _caex(rng: random.Random) -> str:
    return f"CAEX-{rng.randrange(1, 40):02d}"


def _shift(rng: random.Random) -> str:
    return f"Turno {rng.choice('ABCD')}"


def _item(domain: str, question: str, contexts: list[str], truth: str) -> dict:
    return {"question": question, "contexts": contexts, "ground_truth": truth, "domain": domain}


# ---------------------------------------------------------------------------- proceso
def p_throughput(rng, distract):
    mill, shift = f"SAG-{rng.randrange(1, 4):02d}", _shift(rng)
    hours = rng.choice([8, 12])
    tons = rng.randrange(8000, 14500, 50) if hours == 12 else rng.randrange(5500, 9500, 50)
    ctx = [f"Reporte de molienda: el molino {mill} procesó {es(tons)} toneladas de mineral durante el {shift} de {hours} horas."]
    if distract:
        ctx.append(f"El molino de bolas BOL-0{rng.randrange(1, 4)} operó {rng.choice([6, 7])} horas por una detención programada.")
    return _item("proceso", f"¿Cuál fue el throughput promedio del molino {mill} durante el {shift}?", ctx,
                 f"El throughput promedio fue de {es(throughput(tons, hours), 1)} t/h ({es(tons)} t en {hours} h).")


def p_energy(rng, distract):
    mill = f"SAG-{rng.randrange(1, 4):02d}"
    tons, mwh = rng.randrange(9000, 14000, 100), rng.randrange(95, 160)
    ctx = [f"Durante la jornada, el molino {mill} consumió {es(mwh)} MWh de energía eléctrica y procesó {es(tons)} t de mineral."]
    if distract:
        ctx.append("La subestación principal reportó una tensión estable de 13,8 kV en toda la jornada.")
    return _item("proceso", f"¿Cuál fue el consumo de energía específica del molino {mill} en kWh por tonelada?", ctx,
                 f"La energía específica fue de {es(specific_energy_kwh_t(mwh, tons), 1)} kWh/t ({es(mwh)} MWh sobre {es(tons)} t).")


def p_power_change(rng, distract):
    mill = f"BOL-{rng.randrange(1, 4):02d}"
    before = rng.randrange(6000, 8000, 50)
    after = before + rng.choice([-1, 1]) * rng.randrange(200, 900, 50)
    ctx = [f"Telemetría del molino de bolas {mill}: la potencia al eje pasó de {es(before)} kW a las 08:00 a {es(after)} kW a las 14:00."]
    if distract:
        ctx.append(f"El molino SAG-0{rng.randrange(1, 4)} mantuvo su potencia en {es(rng.randrange(9000, 11000, 50))} kW durante el mismo periodo.")
    c = pct_change(before, after)
    word = "aumentó" if c > 0 else "disminuyó"
    return _item("proceso", f"¿En qué porcentaje cambió la potencia del molino {mill} entre las 08:00 y las 14:00?", ctx,
                 f"La potencia {word} {es(abs(c), 1)}% (de {es(before)} kW a {es(after)} kW).")


def p_p80(rng, distract):
    target = rng.choice([150, 160, 170, 180])
    measured = target + rng.randrange(-20, 45, 5)
    line = rng.randrange(1, 4)
    ctx = [f"Control granulométrico de la línea {line}: el P80 medido en el rebose de los hidrociclones fue de {measured} µm; la meta de operación es {target} µm."]
    if distract:
        ctx.append(f"El análisis de la línea {line % 3 + 1} quedó pendiente por falla del analizador en línea.")
    over = pct_over(measured, target)
    if measured == target:
        truth = f"El P80 está en la meta: {measured} µm."
    else:
        truth = f"El P80 está {es(abs(measured - target))} µm ({es(abs(over), 1)}%) {'sobre' if over > 0 else 'bajo'} la meta de {target} µm."
    return _item("proceso", f"¿Cuánto se desvió el P80 de la línea {line} respecto de su meta?", ctx, truth)


def p_circulating(rng, distract):
    fresh = rng.randrange(900, 1300, 10)
    under = int(fresh * rng.choice([2.2, 2.5, 2.8, 3.1]))
    line = rng.randrange(1, 5)
    ctx = [f"Circuito de molienda de bolas de la línea {line}: alimentación fresca de {es(fresh)} t/h y flujo de underflow de los hidrociclones de {es(under)} t/h."]
    if distract:
        ctx.append("La densidad de pulpa en el cajón de bombas se mantuvo en 1,45 t/m3.")
    return _item("proceso", f"¿Cuál fue la carga circulante del circuito de la línea {line}, como porcentaje de la alimentación fresca?", ctx,
                 f"La carga circulante fue de {es(circulating_load_pct(under, fresh), 0)}% ({es(under)} t/h sobre {es(fresh)} t/h).")


def p_recovery(rng, distract):
    feed = rng.randrange(8000, 12000, 100)
    grade = rng.choice([0.55, 0.62, 0.70, 0.78, 0.85])
    rec = rng.choice([82.0, 85.5, 88.0, 90.5])
    fine = round(feed * grade / 100.0 * rec / 100.0, 1)
    day = date(2026, 3, 1) + timedelta(days=rng.randrange(0, 90))
    ctx = [f"Balance metalúrgico del {es_date(day)}: se alimentaron {es(feed)} t con una ley de cobre de {es(grade, 2)}% y se obtuvieron {es(fine, 1)} t de cobre fino en concentrado."]
    if distract:
        ctx.append("La ley de molibdeno del concentrado fue de 0,03%, bajo el límite comercial.")
    return _item("proceso", f"¿Cuál fue la recuperación de cobre del {es_date(day)}?", ctx,
                 f"La recuperación fue de {es(recovery_pct(feed, grade, fine), 1)}% ({es(fine, 1)} t de cobre fino sobre {es(feed * grade / 100.0, 1)} t contenidas).")


# ----------------------------------------------------------------------- mantenimiento
def m_mtbf(rng, distract):
    truck = _caex(rng)
    hours, fails = rng.randrange(1800, 4200, 20), rng.randrange(3, 9)
    ctx = [f"Historial del camión {truck} en el último trimestre: {es(hours)} horas operadas y {fails} fallas que detuvieron el equipo."]
    if distract:
        ctx.append(f"El camión {_caex(rng)} estuvo fuera de servicio por una reparación de tolva.")
    return _item("mantenimiento", f"¿Cuál es el MTBF del camión {truck} en ese periodo?", ctx,
                 f"El MTBF fue de {es(mtbf(hours, fails), 1)} horas ({es(hours)} h operadas con {fails} fallas).")


def m_availability(rng, distract):
    truck, month_hours = _caex(rng), 720
    down = rng.randrange(40, 160, 2)
    ctx = [f"En el mes (720 horas de calendario) el camión {truck} acumuló {down} horas de mantención programada y no programada."]
    if distract:
        ctx.append(f"La flota completa promedió una disponibilidad física de {es(rng.choice([86.5, 88.0, 89.5]), 1)}% en el mismo mes.")
    return _item("mantenimiento", f"¿Cuál fue la disponibilidad física del camión {truck} en el mes?", ctx,
                 f"La disponibilidad fue de {es(availability_pct(month_hours, down), 1)}% ({down} h de detención sobre {month_hours} h).")


def m_next_pm(rng, distract):
    truck = _caex(rng)
    interval = rng.choice([250, 500])
    last = rng.randrange(8000, 15000, interval)
    now = last + rng.randrange(40, interval - 20, 10)
    ctx = [f"El camión {truck} tiene un horómetro de {es(now)} h. Su último mantenimiento preventivo (PM) fue a las {es(last)} h y el intervalo de PM es de {interval} h."]
    if distract:
        ctx.append(f"El camión {_caex(rng)} tiene su PM de {interval} h vencido hace {rng.randrange(10, 60)} horas.")
    due = last + interval
    return _item("mantenimiento", f"¿A qué horómetro corresponde el próximo PM del camión {truck} y cuántas horas faltan?", ctx,
                 f"El próximo PM corresponde a las {es(due)} h y faltan {es(due - now)} h.")


def m_temperature(rng, distract):
    truck = _caex(rng)
    limit = rng.choice([100, 105, 110])
    reading = limit + rng.randrange(2, 14)
    ctx = [f"Alarma de telemetría del camión {truck}: temperatura de aceite de motor de {reading} °C. El límite de alarma configurado es {limit} °C."]
    if distract:
        ctx.append(f"La temperatura del refrigerante del camión {_caex(rng)} está normal, en {rng.randrange(82, 92)} °C.")
    return _item("mantenimiento", f"¿Cuántos grados sobre el límite de alarma estaba la temperatura de aceite del camión {truck}?", ctx,
                 f"Estaba {reading - limit} °C sobre el límite de alarma ({reading} °C frente a {limit} °C).")


def m_backlog(rng, distract):
    truck = _caex(rng)
    a, b = rng.randrange(6, 30), rng.randrange(8, 40)
    ctx = [f"Backlog del camión {truck}: la OT de cambio de bomba hidráulica requiere {a} horas hombre y la OT de reparación del sistema de frenos requiere {b} horas hombre."]
    if distract:
        ctx.append(f"La OT de lavado de {_caex(rng)} se estimó en {rng.randrange(2, 6)} horas hombre.")
    return _item("mantenimiento", f"¿Cuántas horas hombre suman las dos órdenes de trabajo pendientes del camión {truck}?", ctx,
                 f"Suman {a + b} horas hombre ({a} h de la bomba hidráulica y {b} h del sistema de frenos).")


def m_remaining_life(rng, distract):
    truck = _caex(rng)
    life = rng.choice([12000, 15000, 18000, 20000])
    used = rng.randrange(int(life * 0.55), int(life * 0.95), 100)
    ctx = [f"El tren de rodado delantero del camión {truck} tiene una vida útil de diseño de {es(life)} horas y acumula {es(used)} horas desde su instalación."]
    if distract:
        ctx.append(f"Los neumáticos del camión {_caex(rng)} se cambiaron hace {rng.randrange(2000, 5000, 100)} horas.")
    return _item("mantenimiento", f"¿Cuántas horas de vida útil le quedan al tren de rodado delantero del camión {truck} y qué porcentaje se ha consumido?", ctx,
                 f"Le quedan {es(life - used)} horas y se ha consumido el {es(used / life * 100, 1)}% de su vida útil.")


# --------------------------------------------------------------------------- seguridad
def s_frequency(rng, distract):
    accidents = rng.randrange(1, 5)
    man_hours = rng.randrange(450_000, 900_000, 5000)
    month = rng.choice(["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto"])
    ctx = [f"Reporte de seguridad de {month}: {plural(accidents, 'accidente', 'accidentes')} con tiempo perdido y {es(man_hours)} horas hombre trabajadas."]
    if distract:
        ctx.append(f"Se registraron además {rng.randrange(3, 12)} incidentes sin lesión.")
    return _item("seguridad", f"¿Cuál fue el índice de frecuencia de {month} (accidentes con tiempo perdido por millón de horas hombre)?", ctx,
                 f"El índice de frecuencia fue de {es(frequency_index(accidents, man_hours), 2)} ({plural(accidents, 'accidente', 'accidentes')} en {es(man_hours)} horas hombre).")


def s_ramp_limit(rng, distract):
    slope, loaded, empty = rng.choice([8, 10, 12]), rng.choice([15, 20, 25]), rng.choice([30, 35, 40])
    ctx = [f"Procedimiento interno (ilustrativo): en rampas con pendiente de {slope}% o más, la velocidad máxima de los camiones cargados es {loaded} km/h y la de los camiones vacíos es {empty} km/h."]
    if distract:
        ctx.append("En las zonas de carguío la velocidad máxima de todo vehículo es de 20 km/h.")
    return _item("seguridad", f"¿Cuál es la velocidad máxima de un camión cargado en una rampa con pendiente de {slope}%?", ctx,
                 f"La velocidad máxima de un camión cargado es {loaded} km/h.")


def s_co(rng, distract):
    limit = rng.choice([25, 30, 35])
    reading = limit + rng.randrange(3, 30)
    area = rng.choice(["galería norte", "chancador primario", "taller de mantención", "sala de bombas"])
    ctx = [f"Alerta de gases: el detector del {area} marcó {reading} ppm de monóxido de carbono; el límite de exposición definido para el turno es {limit} ppm y el procedimiento indica evacuar el área al superarlo."]
    if distract:
        ctx.append("El detector de la bodega de insumos marcó 4 ppm, sin alteración.")
    return _item("seguridad", f"¿Cuánto superó el CO del {area} el límite del turno y qué indica el procedimiento?", ctx,
                 f"Superó el límite en {reading - limit} ppm ({reading} ppm frente a {limit} ppm) y el procedimiento indica evacuar el área.")


def s_shift_events(rng, distract):
    shift = _shift(rng)
    no_injury, near_miss, first_aid = rng.randrange(0, 4), rng.randrange(1, 6), rng.randrange(0, 3)
    ctx = [f"Reporte de {shift}: {plural(no_injury, 'incidente sin lesión', 'incidentes sin lesión')}, {plural(near_miss, 'cuasi accidente', 'cuasi accidentes')} y {plural(first_aid, 'atención de primeros auxilios', 'atenciones de primeros auxilios')}."]
    if distract:
        ctx.append(f"El día anterior se registraron {rng.randrange(1, 9)} eventos de seguridad en toda la faena.")
    total = no_injury + near_miss + first_aid
    return _item("seguridad", f"¿Cuántos eventos de seguridad se registraron en total durante el {shift}?", ctx,
                 f"Se registraron {total} eventos ({plural(no_injury, 'incidente sin lesión', 'incidentes sin lesión')}, {plural(near_miss, 'cuasi accidente', 'cuasi accidentes')} y {plural(first_aid, 'atención de primeros auxilios', 'atenciones de primeros auxilios')}).")


def s_min_distance(rng, distract):
    vehicle = rng.choice(["vehículo liviano", "camioneta de supervisión", "bus de personal"])
    metres = rng.choice([50, 80, 100, 150])
    ctx = [f"Norma de tránsito de mina (ilustrativa): la distancia mínima de aproximación entre un {vehicle} y un camión CAEX en movimiento es de {metres} metros."]
    if distract:
        ctx.append("La distancia mínima entre dos camiones CAEX en una rampa es de 60 metros.")
    return _item("seguridad", f"¿Qué distancia mínima debe mantener un {vehicle} respecto de un camión CAEX en movimiento?", ctx,
                 f"Debe mantener al menos {metres} metros.")


def s_days_without(rng, distract):
    last = date(2026, 1, 1) + timedelta(days=rng.randrange(0, 120))
    today = last + timedelta(days=rng.randrange(10, 90))
    ctx = [f"Tablero de seguridad: el último accidente con tiempo perdido ocurrió el {es_date(last)}. La fecha del reporte es el {es_date(today)}."]
    if distract:
        ctx.append("El último cuasi accidente con potencial alto se informó la semana pasada.")
    return _item("seguridad", f"¿Cuántos días lleva la faena sin accidentes con tiempo perdido al {es_date(today)}?", ctx,
                 f"Lleva {(today - last).days} días sin accidentes con tiempo perdido.")


FAMILIES = {
    "proceso": [p_throughput, p_energy, p_power_change, p_p80, p_circulating, p_recovery],
    "mantenimiento": [m_mtbf, m_availability, m_next_pm, m_temperature, m_backlog, m_remaining_life],
    "seguridad": [s_frequency, s_ramp_limit, s_co, s_shift_events, s_min_distance, s_days_without],
}


def generate(seed: int = SEED, per_family: int = PER_FAMILY) -> list[dict]:
    """Pares de pregunta, contextos y respuesta; mismos resultados para la misma semilla."""
    rng = random.Random(seed)
    items, seen = [], set()
    for domain in DOMAINS:
        for family in FAMILIES[domain]:
            made, attempts = 0, 0
            while made < per_family:
                attempts += 1
                if attempts > 200:
                    raise RuntimeError(f"{family.__name__} no logra generar {per_family} preguntas distintas")
                item = family(rng, distract=(made % 2 == 1))
                if item["question"] in seen:  # preguntas distintas entre si
                    continue
                seen.add(item["question"])
                items.append(item)
                made += 1
    return items


def export(path: Path = OUT_PATH, seed: int = SEED) -> list[dict]:
    items = generate(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return items


if __name__ == "__main__":
    data = export()
    counts = {d: sum(i["domain"] == d for i in data) for d in DOMAINS}
    print(f"{len(data)} pares escritos en {OUT_PATH}: {counts}")

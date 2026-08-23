"""Generador de reportes narrativos sinteticos de incidentes de seguridad minera.

Produce narrativas realistas en espanol con jerga operacional chilena
(CAEX, chancador, frente de avance, acunadura, EPP, tronadura, pertiga)
para 12 tipos de incidente, cada uno con una distribucion de severidad
propia (LEVE / GRAVE / FATAL) que refleja los criterios reales del
Articulo 77 del DS 132 (ver `data/ds132_sernageomin.txt`).

Cada registro incluye ademas `relevant_articles`: los articulos del DS 132
que un sistema RAG deberia recuperar para ese incidente. Esta etiqueta no
se usa para entrenar el clasificador de severidad (evitaria fuga de
informacion si se derivara del texto), pero sirve como ground truth para
evaluar la calidad del retriever en `tests/`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

SEVERITY_LEVELS = ["LEVE", "GRAVE", "FATAL"]

FAENAS = [
    "Radomiro Tomic",
    "Chuquicamata",
    "El Teniente",
    "Los Bronces",
    "Escondida",
    "Andina",
    "Los Pelambres",
]
TURNOS = ["turno día", "turno noche", "turno A", "turno B", "turno de relevo"]

UBICACIONES_SUBTERRANEAS = [
    "Nivel 3200",
    "Nivel 2850",
    "Rampa Norte",
    "Frente de avance Oeste",
    "Galería 12",
    "Pique de Ventilación Sur",
    "Cámara de Producción 4",
    "Chimenea de traspaso",
]
UBICACIONES_RAJO = [
    "Rajo Fase 4",
    "Banco 1780",
    "Sector Botadero Este",
    "Zona de Carguío Principal",
    "Rampa de acceso al rajo",
    "Polígono de tronadura B-12",
]
UBICACIONES_PLANTA = [
    "Chancador Primario N°2",
    "Correa transportadora CV-14",
    "Planta de Chancado Secundario",
    "Sala de Molienda",
    "Patio de acopio de mineral",
]

TRABAJADOR_ROLES = [
    "operador de perforadora",
    "acuñador",
    "operador de CAEX",
    "mecánico de mantención",
    "supervisor de turno",
    "ayudante de tronadura",
    "operador de chancador",
    "prevencionista de riesgos",
]


def _equipo_id(prefix: str, rng: random.Random) -> str:
    return f"{prefix}-{rng.randint(101, 499)}"


def _pick(pool: list[str], rng: random.Random) -> str:
    return rng.choice(pool)


# Cada tipo de incidente: plantillas narrativas, pesos de severidad y los
# articulos del DS 132 (ver data/ds132_sernageomin.txt) que le son
# aplicables -- usados como ground truth de evaluacion del RAG, no como
# feature del clasificador.
INCIDENT_TYPES: dict[str, dict] = {
    "caida_rocas": {
        "templates": [
            (
                "Durante el {turno} en {faena}, un {rol} que realizaba labores en {ubic} "
                "fue impactado por un desprendimiento de roca no controlado. La inspección "
                "posterior determinó que el sector no contaba con acuñadura reciente y la "
                "fortificación presentaba anomalías no reportadas en el turno anterior."
            ),
            (
                "En {ubic} de la faena {faena}, durante el {turno}, se produjo un derrumbe "
                "parcial de la caja de galería sobre un {rol}. No se había realizado el "
                "chequeo de techo y cajas al inicio de la jornada según el procedimiento de "
                "acuñadura vigente."
            ),
        ],
        "ubicaciones": UBICACIONES_SUBTERRANEAS,
        "severity_weights": {"LEVE": 0.25, "GRAVE": 0.5, "FATAL": 0.25},
        "relevant_articles": ["157", "158", "160", "161", "162"],
    },
    "colapso_fortificacion": {
        "templates": [
            (
                "En el pique de tránsito de {faena}, la fortificación cedió durante el "
                "{turno} mientras un {rol} realizaba mantención en {ubic}. La última "
                "revisión de fortificación registrada superaba el plazo de seis meses "
                "establecido para piques fortificados."
            ),
        ],
        "ubicaciones": UBICACIONES_SUBTERRANEAS,
        "severity_weights": {"LEVE": 0.15, "GRAVE": 0.45, "FATAL": 0.4},
        "relevant_articles": ["157", "158", "159", "160"],
    },
    "atropello_caex": {
        "templates": [
            (
                "En {faena}, durante el {turno}, un vehículo liviano que transitaba por "
                "{ubic} sin pértiga reglamentaria fue impactado por un CAEX {equipo} de "
                "gran tonelaje que realizaba maniobras de retroceso en la zona de carguío. "
                "El operador del CAEX reportó visibilidad reducida por polvo en suspensión."
            ),
            (
                "Un {rol} a pie fue atropellado por el CAEX {equipo} en {ubic} de la faena "
                "{faena}, durante el {turno}. El trabajador no portaba elementos de alta "
                "visibilidad al ingresar al área de operación del rajo."
            ),
        ],
        "ubicaciones": UBICACIONES_RAJO,
        "severity_weights": {"LEVE": 0.1, "GRAVE": 0.35, "FATAL": 0.55},
        "relevant_articles": ["246", "247"],
    },
    "atrapamiento_chancador": {
        "templates": [
            (
                "Un {rol} sufrió atrapamiento parcial al intentar destrabar mineral "
                "sobredimensionado en {ubic} de {faena}, durante el {turno}, con el equipo "
                "aún energizado. No se aplicó el procedimiento de bloqueo y etiquetado "
                "previo a la intervención."
            ),
        ],
        "ubicaciones": UBICACIONES_PLANTA,
        "severity_weights": {"LEVE": 0.1, "GRAVE": 0.4, "FATAL": 0.5},
        "relevant_articles": ["78"],
    },
    "exposicion_polvo_silice": {
        "templates": [
            (
                "Trabajadores del {turno} en {ubic} de {faena} presentaron síntomas "
                "respiratorios tras una jornada de perforación sin riego de agua previo ni "
                "posterior a la tronadura. Los registros de muestreo periódico de aire no "
                "estaban actualizados."
            ),
            (
                "Un {rol} reportó irritación respiratoria aguda luego de operar sin EPP "
                "respiratorio adecuado en {ubic}, {faena}, durante el {turno}, en un sector "
                "con alta concentración de polvo en suspensión."
            ),
        ],
        "ubicaciones": UBICACIONES_RAJO + UBICACIONES_PLANTA,
        "severity_weights": {"LEVE": 0.7, "GRAVE": 0.28, "FATAL": 0.02},
        "relevant_articles": ["140"],
    },
    "tronadura_prematura": {
        "templates": [
            (
                "Durante el carguío de explosivos en {ubic} de {faena}, {turno}, se produjo "
                "una detonación prematura mientras el equipo mecanizado de tapado de hoyos "
                "operaba a menos de veinte metros de la zona de carguío. Un {rol} resultó "
                "afectado por la onda expansiva."
            ),
            (
                "Se registró una tronadura no programada en {ubic}, faena {faena}, durante "
                "el {turno}, en condiciones de viento superior a cien kilómetros por hora "
                "que no fueron evaluadas antes de continuar el carguío de explosivos."
            ),
        ],
        "ubicaciones": UBICACIONES_RAJO,
        "severity_weights": {"LEVE": 0.05, "GRAVE": 0.35, "FATAL": 0.6},
        "relevant_articles": ["250", "251", "252", "253"],
    },
    "caida_altura": {
        "templates": [
            (
                "Un {rol} cayó desde una plataforma suspendida en {ubic} de {faena}, "
                "durante el {turno}. El arnés utilizado no correspondía a un elemento de "
                "protección personal certificado según el procedimiento de trabajos en "
                "altura."
            ),
        ],
        "ubicaciones": UBICACIONES_PLANTA + UBICACIONES_SUBTERRANEAS,
        "severity_weights": {"LEVE": 0.2, "GRAVE": 0.45, "FATAL": 0.35},
        "relevant_articles": ["174"],
    },
    "intoxicacion_gases": {
        "templates": [
            (
                "En {ubic} de la mina subterránea {faena}, durante el {turno}, un {rol} "
                "presentó síntomas de intoxicación tras ingresar a un sector con "
                "concentración de oxígeno bajo el mínimo reglamentario. El aforo de "
                "ventilación trimestral del sector estaba atrasado."
            ),
        ],
        "ubicaciones": UBICACIONES_SUBTERRANEAS,
        "severity_weights": {"LEVE": 0.3, "GRAVE": 0.45, "FATAL": 0.25},
        "relevant_articles": ["137", "138", "139", "144"],
    },
    "incendio_equipo": {
        "templates": [
            (
                "Se declaró un incendio en el sistema hidráulico del CAEX {equipo} en "
                "{ubic}, faena {faena}, durante el {turno}. El equipo contaba con extintor "
                "vencido y el operador no logró controlar el foco inicial."
            ),
        ],
        "ubicaciones": UBICACIONES_RAJO,
        "severity_weights": {"LEVE": 0.4, "GRAVE": 0.45, "FATAL": 0.15},
        "relevant_articles": ["77"],
    },
    "electrocucion": {
        "templates": [
            (
                "Un {rol} sufrió contacto eléctrico al realizar labores de acuñadura cerca "
                "de conductores energizados en {ubic} de {faena}, durante el {turno}. Los "
                "conductores no habían sido desenergizados previo al inicio de la tarea."
            ),
        ],
        "ubicaciones": UBICACIONES_SUBTERRANEAS,
        "severity_weights": {"LEVE": 0.15, "GRAVE": 0.4, "FATAL": 0.45},
        "relevant_articles": ["163"],
    },
    "proyeccion_particulas": {
        "templates": [
            (
                "Durante labores de perforación en {ubic} de {faena}, {turno}, un {rol} "
                "resultó afectado por proyección de esquirlas de roca. El área no contaba "
                "con delimitación ni el trabajador usaba protección facial adecuada."
            ),
        ],
        "ubicaciones": UBICACIONES_RAJO + UBICACIONES_SUBTERRANEAS,
        "severity_weights": {"LEVE": 0.55, "GRAVE": 0.4, "FATAL": 0.05},
        "relevant_articles": ["174"],
    },
    "quemadura_quimica": {
        "templates": [
            (
                "Un {rol} sufrió quemaduras al manipular reactivos en {ubic} de la planta "
                "{faena}, durante el {turno}, sin el EPP químico especificado en la hoja de "
                "seguridad del insumo."
            ),
        ],
        "ubicaciones": UBICACIONES_PLANTA,
        "severity_weights": {"LEVE": 0.35, "GRAVE": 0.55, "FATAL": 0.1},
        "relevant_articles": ["77"],
    },
}

EQUIPO_PREFIX_BY_TYPE = {
    "atropello_caex": "CAEX",
    "incendio_equipo": "CAEX",
}

# Frases de consecuencia por severidad, redactadas segun los criterios reales
# del Articulo 77 del DS 132 (fractura, amputacion, ceguera/sordera total,
# quemaduras invalidantes, intoxicacion masiva -> GRAVE; muerte -> FATAL).
# Sin esto, la severidad quedaba asignada al azar sin ninguna senal en el
# texto -- el clasificador de severidad no tenia nada real que aprender.
CONSEQUENCE_PHRASES: dict[str, list[str]] = {
    "LEVE": [
        "El trabajador presentó una contusión leve y fue dado de alta tras la atención de "
        "primeros auxilios, sin tiempo perdido.",
        "Se registró una lesión menor sin incapacidad laboral, atendida en la posta de la "
        "faena.",
        "No hubo lesiones de consideración; el trabajador continuó su turno tras evaluación "
        "médica preventiva.",
        "El incidente no generó lesiones a personas, solo daños menores a equipos.",
    ],
    "GRAVE": [
        "El trabajador sufrió fractura de columna vertebral y fue trasladado de urgencia al "
        "centro asistencial más cercano.",
        "Se produjo amputación parcial de una extremidad superior, con traslado inmediato en "
        "ambulancia.",
        "El trabajador presentó quemaduras de consideración, con riesgo de invalidez parcial.",
        "Se registró una intoxicación masiva que afectó a varios trabajadores del sector.",
        "La lesión presenta alto potencial de invalidez total y permanente, según la "
        "evaluación médica inicial.",
        "El trabajador perdió la visión de un ojo producto del incidente.",
    ],
    "FATAL": [
        "El trabajador falleció en el lugar antes de la llegada de la brigada de rescate.",
        "Se confirmó el deceso del trabajador durante el traslado al centro asistencial.",
        "El hecho resultó en la muerte de un trabajador, activándose el protocolo de "
        "notificación inmediata a la Dirección Regional del Servicio.",
    ],
}


def _sample_severity(weights: dict[str, float], rng: random.Random) -> str:
    return rng.choices(SEVERITY_LEVELS, weights=[weights[s] for s in SEVERITY_LEVELS], k=1)[0]


def _consequence_phrase(severity: str, rng: random.Random) -> str:
    return rng.choice(CONSEQUENCE_PHRASES[severity])


def generate_incident(incident_type: str, config: dict, incident_id: int, rng: random.Random) -> dict:
    template = rng.choice(config["templates"])
    faena = _pick(FAENAS, rng)
    turno = _pick(TURNOS, rng)
    ubic = _pick(config["ubicaciones"], rng)
    rol = _pick(TRABAJADOR_ROLES, rng)
    equipo = _equipo_id(EQUIPO_PREFIX_BY_TYPE.get(incident_type, "EQ"), rng)

    base_narrative = template.format(turno=turno, faena=faena, ubic=ubic, rol=rol, equipo=equipo)
    severity = _sample_severity(config["severity_weights"], rng)
    narrative = f"{base_narrative} {_consequence_phrase(severity, rng)}"

    return {
        "incident_id": f"INC-{incident_id:04d}",
        "narrative_text": narrative,
        "incident_type": incident_type,
        "severity": severity,
        "faena": faena,
        "turno": turno,
        "ubicacion": ubic,
        "relevant_articles": config["relevant_articles"],
    }


def generate_dataset(n_per_type: int = 12, seed: int = 42) -> list[dict]:
    """Genera `n_per_type` incidentes por cada uno de los 12 tipos (>=100 en total)."""
    rng = random.Random(seed)
    records = []
    incident_id = 1
    for incident_type, config in INCIDENT_TYPES.items():
        for _ in range(n_per_type):
            records.append(generate_incident(incident_type, config, incident_id, rng))
            incident_id += 1
    rng.shuffle(records)
    return records


def save_dataset(records: list[dict], path: Path | None = None) -> Path:
    path = path or (DATA_DIR / "raw_incidents.json")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    records = generate_dataset()
    path = save_dataset(records)

    from collections import Counter

    severity_counts = Counter(r["severity"] for r in records)
    type_counts = Counter(r["incident_type"] for r in records)

    print(f"Incidentes generados: {len(records)}")
    print(f"Tipos de incidente: {len(type_counts)}")
    print(f"Distribución de severidad: {dict(severity_counts)}")
    print(f"Guardado en: {path}")
    print("\nEjemplo:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

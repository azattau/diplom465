# -*- coding: utf-8 -*-
"""
Модуль геокодирования по справочнику населённых пунктов Республики Башкортостан (settlments.csv),
разрешения омонимии одноимённых посёлков/деревень в разных районах и расчёта вектора движения.
"""

import csv
import math
import os
import re
from typing import Dict, List, Optional, Tuple

TYPE_PREFIXES = [
    "рабочий поселок",
    "рабочий посёлок",
    "деревня станции",
    "деревня разъезда",
    "деревня разъезд",
    "село станции",
    "поселок станции",
    "посёлок станции",
    "деревня",
    "село",
    "хутор",
    "посёлок",
    "поселок",
    "кордон",
    "город",
    "пгт.",
    "пгт",
    "р.п.",
    "рп.",
    "рп",
    "пос.",
    "дер.",
    "сел.",
    "хут.",
    "г.",
    "с.",
    "д.",
    "п.",
    "х.",
]

COMPASS_AZIMUTHS = {
    "север": 0.0,
    "с": 0.0,
    "северо-восток": 45.0,
    "св": 45.0,
    "восток": 90.0,
    "в": 90.0,
    "юго-восток": 135.0,
    "юв": 135.0,
    "юг": 180.0,
    "ю": 180.0,
    "юго-запад": 225.0,
    "юз": 225.0,
    "запад": 270.0,
    "з": 270.0,
    "северо-запад": 315.0,
    "сз": 315.0,
}


# Разговорные названия населённых пунктов, часто встречающиеся в реальных сводках ПВН 112:
PVN_ALIASES = {
    "раевка": ("раевский", "Альшеевский"),
    "раевку": ("раевский", "Альшеевский"),
    "раевки": ("раевский", "Альшеевский"),
    "раевке": ("раевский", "Альшеевский"),
    "п раевский": ("раевский", "Альшеевский"),
    "верхне троицк": ("верхнетроицкое", "Туймазинский"),
    "верхне троицка": ("верхнетроицкое", "Туймазинский"),
    "верхнетроицк": ("верхнетроицкое", "Туймазинский"),
    "устьикинск": ("большеустьикинское", "Мечетлинский"),
    "устьикинска": ("большеустьикинское", "Мечетлинский"),
    "усть икинск": ("большеустьикинское", "Мечетлинский"),
    "большеустьикинское": ("большеустьикинское", "Мечетлинский"),
    "татышлы": ("верхние татышлы", "Татышлинский"),
    "верхние татышлы": ("верхние татышлы", "Татышлинский"),
    "балтачево": ("старобалтачево", "Балтачевский"),
    "мияки": ("киргиз-мияки", "Миякинский"),
    "красноусольск": ("красноусольский", "Гафурийский"),
    "красноусольска": ("красноусольский", "Гафурийский"),
    "кандрыкуль": ("кандры-куль", "Туймазинский"),
    "кандрыкуля": ("кандры-куль", "Туймазинский"),
    "нижняя чатра": ("нижняя чатра", "Буздякский"),
    "н чатра": ("нижняя чатра", "Буздякский"),
    "приютово": ("приютово", "Белебеевский"),
}

# Внешние ориентиры направления из файла «Сообщения из ПВН.docx» (соседние регионы):
EXTERNAL_DIRECTION_TARGETS = {
    "екатеринбург": {"full_name": "г. Екатеринбург (Свердловская обл.)", "lat": 56.838926, "lon": 60.605702, "district": "Свердловская область"},
    "екб": {"full_name": "г. Екатеринбург (Свердловская обл.)", "lat": 56.838926, "lon": 60.605702, "district": "Свердловская область"},
    "бавлы": {"full_name": "г. Бавлы (Республика Татарстан)", "lat": 54.3931, "lon": 53.2603, "district": "Республика Татарстан"},
    "оренбургская область": {"full_name": "Оренбургская область (южное направление)", "lat": 51.7682, "lon": 55.0970, "district": "Оренбургская область"},
    "челябинская область": {"full_name": "Челябинская область (восточное направление)", "lat": 55.1644, "lon": 61.4368, "district": "Челябинская область"},
}

INVALID_PLACEHOLDERS = {
    "нет", "не указан", "не указано", "неизвестно", "не определен", "не определено",
    "none", "null", "unknown", "-", "—", ""
}


def normalize_text(text: str) -> str:
    """Приводит строку к нижнему регистру, заменяет 'ё' на 'е' и очищает от спецсимволов."""
    if not text:
        return ""
    s = str(text).strip().lower().replace("ё", "е")
    s = re.sub(r"[^а-яa-z0-9\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_district_stem(district_str: str) -> str:
    """
    Выделяет основу названия муниципального района (например:
    'Абзелиловский муниципальный район' -> 'абзелиловск',
    'Уфимского района' -> 'уфимск',
    'в Шаранском р-не' -> 'шаранск').
    """
    norm = normalize_text(district_str)
    if not norm or norm in INVALID_PLACEHOLDERS:
        return ""
    # Убираем служебные слова
    norm = re.sub(r"\b(муниципальный|муниципального|муниципальном|район|района|районе|р-н|р-на|р-не|городской|округ|город)\b", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return ""
    first_word = norm.split()[0]
    # Отсекаем падежные окончания прилагательных (-ий, -ый, -ой, -ого, -ому, -им, -ым, -ом, -ем)
    stem = re.sub(r"(ий|ый|ой|ого|его|ому|ему|им|ым|ом|ем|ая|яя|ую|юю|ое|ее)$", "", first_word)
    return stem if len(stem) >= 3 else first_word


def clean_settlement_name(raw_name: str) -> Tuple[str, str]:
    """
    Отделяет тип населённого пункта (включая сокращения 'г.', 'с.', 'д.', 'п.', 'поселок')
    и скобочные пояснения от собственного названия.
    Возвращает кортеж: (очищенное_название, определенный_тип_нп).
    """
    if not raw_name:
        return "", ""
    s = str(raw_name).strip()
    # Убираем возможные скобки, которые иногда добавляет LLM, например: "Шаран (Шаранский район)"
    s = re.sub(r"\(.*?\)", " ", s).strip()
    s_lower = s.lower().replace("ё", "е")
    s_lower = re.sub(r"\s+", " ", s_lower).strip()

    if s_lower in INVALID_PLACEHOLDERS:
        return "", ""

    detected_type = ""
    for prefix in TYPE_PREFIXES:
        if s_lower.startswith(prefix + " ") or (prefix.endswith(".") and s_lower.startswith(prefix)):
            p_clean = prefix.rstrip(".")
            if p_clean in ("г", "город"):
                detected_type = "город"
            elif p_clean in ("с", "сел", "село", "село станции"):
                detected_type = "село"
            elif p_clean in ("д", "дер", "деревня", "деревня станции", "деревня разъезда", "деревня разъезд"):
                detected_type = "деревня"
            elif p_clean in ("п", "пос", "поселок", "посёлок", "рп", "р.п", "пгт", "рабочий поселок", "поселок станции"):
                detected_type = "поселок"
            elif p_clean in ("х", "хут", "хутор"):
                detected_type = "хутор"
            else:
                detected_type = p_clean
            s_lower = s_lower[len(prefix):].strip()
            break

    # Если после названия написано слово "район" (например, LLM вернула "Шаранский район"), убираем его
    s_lower = re.sub(r"\s+(муниципальный\s+)?(район|р-н).*$", "", s_lower).strip()

    return normalize_text(s_lower), detected_type


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Вычисляет расстояние в километрах между двумя точками."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Вычисляет начальный азимут (от 0 до 360 градусов) из точки 1 в точку 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


def project_point(lat: float, lon: float, bearing_deg: float, distance_km: float = 22.0) -> Tuple[float, float]:
    """Строит конечную точку вектора на заданном расстоянии (км) по азимуту (градусы)."""
    r = 6371.0
    ad = distance_km / r
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ad) + math.cos(lat1) * math.sin(ad) * math.cos(br)
    )
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(ad) * math.cos(lat1),
        math.cos(ad) - math.sin(lat1) * math.sin(lat2),
    )
    return round(math.degrees(lat2), 6), round(math.degrees(lon2), 6)


class SettlementGeocoder:
    """Справочник населённых пунктов с поддержкой разрешения омонимии по районам."""

    def __init__(self, csv_path: str = "settlments.csv"):
        self.csv_path = csv_path
        self.records: List[Dict] = []
        self.districts: List[str] = []
        self.district_stems: Dict[str, str] = {}
        self._load_csv()

    def _load_csv(self) -> None:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Справочник не найден: {self.csv_path}")

        districts_set = set()
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_full = (row.get("Населённый пункт") or "").strip()
                raw_type = (row.get("Тип НП") or "").strip().lower().replace("ё", "е")
                district = (row.get("Муниципальный район / городской округ") or "").strip()
                municipality = (row.get("Муниципальное образование") or "").strip()

                try:
                    lat = float(row.get("Широта"))
                    lon = float(row.get("Долгота"))
                except (TypeError, ValueError):
                    continue

                clean_name, inferred_type = clean_settlement_name(raw_full)
                if not clean_name:
                    continue

                final_type = raw_type if raw_type else inferred_type
                if raw_full.lower().startswith("г.") or "городской округ" in district.lower():
                    if not final_type:
                        final_type = "город"

                nospace_name = clean_name.replace(" ", "").replace("-", "")
                dist_stem = extract_district_stem(district)
                if district:
                    districts_set.add(district)
                    if dist_stem:
                        self.district_stems[dist_stem] = district

                self.records.append({
                    "id": str(row.get("№", "")).strip(),
                    "full_name": raw_full,
                    "clean_name": clean_name,
                    "nospace_name": nospace_name,
                    "type": final_type or "нп",
                    "district": district,
                    "district_norm": normalize_text(district),
                    "district_stem": dist_stem,
                    "municipality": municipality,
                    "lat": lat,
                    "lon": lon,
                    "fias_id": (row.get("FIAS ID") or "").strip(),
                })

        self.districts = sorted(districts_set)

    def find_candidates(
        self,
        name_nominative: str,
        settlement_type: str = "",
        district_hint: str = "",
        reference_coords: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Optional[Dict], List[Dict], bool]:
        """
        Ищет населённый пункт в справочнике.
        Возвращает:
          (лучший_кандидат, список_всех_одноименных_кандидатов_в_разных_районах, требуется_ли_уточнение_района)
        """
        clean_query, prefix_type = clean_settlement_name(name_nominative)
        if not clean_query:
            return None, [], False

        
        if clean_query in PVN_ALIASES:
            alias_name, alias_dist = PVN_ALIASES[clean_query]
            clean_query = alias_name
            if not district_hint:
                district_hint = alias_dist
        if clean_query in EXTERNAL_DIRECTION_TARGETS:
            ext_t = EXTERNAL_DIRECTION_TARGETS[clean_query]
            ext_rec = {
                "id": "ext_" + clean_query,
                "full_name": ext_t["full_name"],
                "clean_name": clean_query,
                "nospace_name": clean_query.replace(" ", ""),
                "type": "ориентир",
                "district": ext_t["district"],
                "district_norm": ext_t["district"].lower(),
                "district_stem": "",
                "municipality": "",
                "lat": ext_t["lat"],
                "lon": ext_t["lon"],
                "fias_id": "",
            }
            return ext_rec, [ext_rec], False
        query_nospace = clean_query.replace(" ", "").replace("-", "")
        type_hint = normalize_text(settlement_type or prefix_type)
        dist_stem = extract_district_stem(district_hint)

        # 1. Точное совпадение по очищенному имени
        candidates = [
            r for r in self.records
            if r["clean_name"] == clean_query or r["nospace_name"] == query_nospace
        ]

        # 2. Если точного совпадения нет, ищем по морфологической основе слова
        if not candidates and len(query_nospace) >= 3:
            stem = query_nospace[:-2] if (len(query_nospace) > 5 and query_nospace.endswith(("ово", "ево", "ино", "ыно", "ский", "цкий", "ая", "ое", "ий", "ый"))) else (
                query_nospace[:-1] if (len(query_nospace) >= 4 and query_nospace[-1] in "аяоеиыуьюй") else query_nospace
            )
            if len(stem) >= 3:
                candidates = [
                    r for r in self.records
                    if r["nospace_name"] == stem
                    or (r["nospace_name"].startswith(stem) and abs(len(r["nospace_name"]) - len(query_nospace)) <= 2)
                ]

        if not candidates:
            return None, [], False

        if len(candidates) == 1:
            return candidates[0], candidates, False

        # Проверяем, фильтруется ли список однозначно по указанному району
        district_matched = []
        if dist_stem and len(dist_stem) >= 3:
            district_matched = [
                c for c in candidates
                if dist_stem in c["district_norm"] or (c["district_stem"] and c["district_stem"].startswith(dist_stem[:4]))
            ]

        # Ранжируем всех кандидатов
        scored = []
        for cand in candidates:
            score = 0.0
            if dist_stem and len(dist_stem) >= 3:
                if dist_stem in cand["district_norm"] or (cand["district_stem"] and cand["district_stem"].startswith(dist_stem[:4])):
                    score += 100.0

            if type_hint:
                if type_hint in cand["type"] or (type_hint in ("г", "город") and cand["type"] == "город"):
                    score += 35.0
                elif type_hint in ("поселок", "посёлок", "село") and cand["type"] in ("село", "поселок"):
                    score += 15.0
            else:
                if cand["type"] == "город":
                    score += 40.0
                elif cand["type"] == "село":
                    score += 8.0

            # Если название НП однокоренное с названием самого района (например, село Шаран в Шаранском районе,
            # село Чекмагуш в Чекмагушевском районе), даём бонус районному центру!
            if len(cand["nospace_name"]) >= 4 and cand["district_norm"].startswith(cand["nospace_name"][:4]):
                score += 20.0

            if reference_coords is not None:
                dist_km = haversine_km(cand["lat"], cand["lon"], reference_coords[0], reference_coords[1])
                score -= dist_km * 0.05

            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        ordered_candidates = [item[1] for item in scored]
        best = ordered_candidates[0]

        # Определяем, нужна ли подсказка выбора района оператору:
        # Если кандидатов > 1 в разных районах, это не крупный город (как г. Белорецк / г. Октябрьский),
        # и район НЕ был явно указан в сообщении
        unique_districts = {c["district"] for c in candidates}
        is_ambiguous = (
            len(unique_districts) > 1
            and not district_matched
            and best["type"] != "город"
        )

        return best, ordered_candidates, is_ambiguous

    def find_settlement(
        self,
        name_nominative: str,
        settlement_type: str = "",
        district_hint: str = "",
        reference_coords: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict]:
        best, _, _ = self.find_candidates(name_nominative, settlement_type, district_hint, reference_coords)
        return best

    def build_vector(self, seen_rec: Dict, target_rec: Optional[Dict], origin_rec: Optional[Dict], dir_compass: str) -> Optional[Dict]:
        lat1, lon1 = seen_rec["lat"], seen_rec["lon"]
        if target_rec and (target_rec["lat"] != lat1 or target_rec["lon"] != lon1):
            bearing = calculate_bearing(lat1, lon1, target_rec["lat"], target_rec["lon"])
            dist_to_target = haversine_km(lat1, lon1, target_rec["lat"], target_rec["lon"])
            arrow_km = min(max(dist_to_target * 0.45, 12.0), 35.0)
            v_lat, v_lon = project_point(lat1, lon1, bearing, arrow_km)
            return {
                "has_vector": True,
                "mode": "target_settlement",
                "target_name": target_rec["full_name"],
                "target_district": target_rec["district"],
                "target_lat": target_rec["lat"],
                "target_lon": target_rec["lon"],
                "bearing_deg": round(bearing, 1),
                "distance_to_target_km": round(dist_to_target, 1),
                "arrow_end_lat": v_lat,
                "arrow_end_lon": v_lon,
                "description": f"В сторону: {target_rec['full_name']} (азимут {round(bearing)}°, ~{round(dist_to_target)} км)",
            }
        elif dir_compass in COMPASS_AZIMUTHS:
            bearing = COMPASS_AZIMUTHS[dir_compass]
            v_lat, v_lon = project_point(lat1, lon1, bearing, 22.0)
            return {
                "has_vector": True,
                "mode": "compass",
                "target_name": f"на {dir_compass}",
                "target_lat": v_lat,
                "target_lon": v_lon,
                "bearing_deg": round(bearing, 1),
                "distance_to_target_km": 22.0,
                "arrow_end_lat": v_lat,
                "arrow_end_lon": v_lon,
                "description": f"Курс: на {dir_compass} (азимут {round(bearing)}°)",
            }
        elif origin_rec and (origin_rec["lat"] != lat1 or origin_rec["lon"] != lon1):
            bearing = calculate_bearing(origin_rec["lat"], origin_rec["lon"], lat1, lon1)
            v_lat, v_lon = project_point(lat1, lon1, bearing, 22.0)
            return {
                "has_vector": True,
                "mode": "from_origin",
                "target_name": f"курс от {origin_rec['full_name']}",
                "target_lat": v_lat,
                "target_lon": v_lon,
                "bearing_deg": round(bearing, 1),
                "distance_to_target_km": 22.0,
                "arrow_end_lat": v_lat,
                "arrow_end_lon": v_lon,
                "description": f"Со стороны: {origin_rec['full_name']} (азимут {round(bearing)}°)",
            }
        return None

    def resolve_observation(self, extracted: Dict, forced_settlement_id: str = "") -> Dict:
        """
        Сопоставляет результат гибридного парсера со справочником settlments.csv.
        Поддерживает принудительный выбор конкретного НП по ID (когда оператор уточняет район).
        """
        seen_name = extracted.get("settlement_seen") or ""
        seen_type = extracted.get("settlement_type") or ""
        district = extracted.get("district") or ""
        dir_to_name = extracted.get("direction_to") or ""
        dir_from_name = extracted.get("direction_from") or ""
        dir_compass = normalize_text(extracted.get("direction_compass") or "")

        target_rec = self.find_settlement(dir_to_name) if dir_to_name else None
        origin_rec = self.find_settlement(dir_from_name) if dir_from_name else None

        ref_coords = None
        if target_rec:
            ref_coords = (target_rec["lat"], target_rec["lon"])
        elif origin_rec:
            ref_coords = (origin_rec["lat"], origin_rec["lon"])

        seen_rec, all_candidates, is_ambiguous = self.find_candidates(
            seen_name,
            settlement_type=seen_type,
            district_hint=district,
            reference_coords=ref_coords,
        )

        # Если оператор явно кликнул на конкретный район из списка одноимённых НП
        if forced_settlement_id:
            for r in self.records:
                if str(r["id"]) == str(forced_settlement_id):
                    seen_rec = r
                    is_ambiguous = False
                    if not all_candidates:
                        _, all_candidates, _ = self.find_candidates(r["clean_name"])
                    break

        # Если точка наблюдения не была выделена отдельно, но распознан один топоним в тексте
        if not seen_rec and target_rec:
            seen_rec, all_candidates, is_ambiguous = self.find_candidates(
                dir_to_name, settlement_type=seen_type, district_hint=district
            )
            target_rec = None

        if not seen_rec and origin_rec:
            seen_rec, all_candidates, is_ambiguous = self.find_candidates(
                dir_from_name, settlement_type=seen_type, district_hint=district
            )
            origin_rec = None

        if not seen_rec:
            return {
                "found": False,
                "error": f"Населённый пункт «{seen_name or 'не распознан'}» не найден в справочнике settlments.csv.",
            }

        vector_info = self.build_vector(seen_rec, target_rec, origin_rec, dir_compass)

        # Формируем компактный список альтернативных районов для одноимённых посёлков/деревень
        alternatives = []
        if len(all_candidates) > 1:
            for c in all_candidates[:15]:
                alternatives.append({
                    "id": c["id"],
                    "full_name": c["full_name"],
                    "district": c["district"],
                    "municipality": c["municipality"],
                    "lat": c["lat"],
                    "lon": c["lon"],
                })

        return {
            "found": True,
            "settlement": seen_rec,
            "vector": vector_info,
            "is_ambiguous": is_ambiguous,
            "total_homonyms": len(all_candidates),
            "alternatives": alternatives,
        }


def parse_hhmm_to_minutes(time_str: str) -> Optional[int]:
    """Переводит время формата 'ЧЧ:ММ' в количество минут от начала суток."""
    if not time_str:
        return None
    m = re.match(r"^\s*([01]?\d|2[0-3])[:.]([0-5]\d)\s*$", str(time_str))
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def angle_diff_deg(a1: float, a2: float) -> float:
    """Минимальная разница между двумя азимутами в градусах (от 0 до 180)."""
    diff = abs((a1 - a2) % 360.0)
    return 360.0 - diff if diff > 180.0 else diff


def compute_track_and_threats(observations: List[Dict], targets: List[Dict], default_speed_kmh: float = 120.0) -> Dict:
    """
    Строит единый трек пролёта по упорядоченным во времени точкам,
    рассчитывает скорость (км/ч), прогнозные точки (+15 и +30 мин)
    и оценивает вероятность угрозы для каждого охраняемого объекта оператора.
    """
    # 1. Сортируем наблюдения по времени ЧЧ:ММ (при равенстве — по порядку добавления)
    valid_obs = []
    for idx, obs in enumerate(observations):
        try:
            lat = float(obs["lat"])
            lon = float(obs["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        mins = parse_hhmm_to_minutes(obs.get("time", ""))
        valid_obs.append((mins if mins is not None else 9999 + idx, idx, obs))

    valid_obs.sort(key=lambda x: (x[0], x[1]))
    ordered = [x[2] for x in valid_obs]

    track_info = {
        "has_track": len(ordered) >= 2,
        "points_count": len(ordered),
        "points": [[o["lat"], o["lon"]] for o in ordered],
        "segments": [],
        "total_distance_km": 0.0,
        "estimated_speed_kmh": default_speed_kmh,
        "speed_source": "крейсерская по умолчанию (120 км/ч)",
        "current_bearing_deg": None,
        "forecast_15m": None,
        "forecast_30m": None,
    }

    # 2. Если точек >= 2, считаем сегменты маршрута и реальную скорость
    total_dist = 0.0
    last_bearing = None
    last_speed = None

    for i in range(len(ordered) - 1):
        p1, p2 = ordered[i], ordered[i + 1]
        dist_km = haversine_km(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        if dist_km < 0.05:
            continue
        br = calculate_bearing(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        t1 = parse_hhmm_to_minutes(p1.get("time", ""))
        t2 = parse_hhmm_to_minutes(p2.get("time", ""))
        dt_min = (t2 - t1) if (t1 is not None and t2 is not None and t2 > t1) else None
        seg_speed = round((dist_km / (dt_min / 60.0)), 1) if (dt_min and dt_min > 0) else None
        if seg_speed and 20.0 <= seg_speed <= 600.0:
            last_speed = seg_speed
        total_dist += dist_km
        last_bearing = br
        track_info["segments"].append({
            "from_name": p1.get("settlement_full", ""),
            "to_name": p2.get("settlement_full", ""),
            "distance_km": round(dist_km, 1),
            "dt_minutes": dt_min,
            "speed_kmh": seg_speed,
            "bearing_deg": round(br, 1),
        })

    track_info["total_distance_km"] = round(total_dist, 1)

    # Определяем текущий курс и последнюю точку для прогноза и оценки угрозы
    active_lat, active_lon = None, None
    active_bearing = None

    if len(ordered) >= 1:
        last_obs = ordered[-1]
        active_lat, active_lon = float(last_obs["lat"]), float(last_obs["lon"])
        # Если у последней точки есть собственный вектор из текста ("в сторону ..."), учитываем его,
        # иначе берём курс последнего сегмента трека
        vec = last_obs.get("vector") or {}
        if vec.get("has_vector") and vec.get("bearing_deg") is not None:
            active_bearing = float(vec["bearing_deg"])
        elif last_bearing is not None:
            active_bearing = float(last_bearing)

    if last_speed is not None:
        track_info["estimated_speed_kmh"] = last_speed
        track_info["speed_source"] = "рассчитана по времени между точками"

    speed_kmh = track_info["estimated_speed_kmh"]

    if active_lat is not None and active_bearing is not None:
        track_info["current_bearing_deg"] = round(active_bearing, 1)
        dist_15 = speed_kmh * 0.25
        dist_30 = speed_kmh * 0.50
        f15_lat, f15_lon = project_point(active_lat, active_lon, active_bearing, dist_15)
        f30_lat, f30_lon = project_point(active_lat, active_lon, active_bearing, dist_30)
        track_info["forecast_15m"] = {
            "lat": f15_lat,
            "lon": f15_lon,
            "distance_km": round(dist_15, 1),
            "label": f"Прогноз +15 мин (~{round(dist_15)} км)",
        }
        track_info["forecast_30m"] = {
            "lat": f30_lat,
            "lon": f30_lon,
            "distance_km": round(dist_30, 1),
            "label": f"Прогноз +30 мин (~{round(dist_30)} км)",
        }

    # 3. Оценка угрозы для каждого охраняемого объекта оператора
    evaluated_targets = []
    for t in targets:
        try:
            t_lat = float(t["lat"])
            t_lon = float(t["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        item = {
            "id": t.get("id"),
            "name": t.get("name", "Охраняемый объект"),
            "lat": t_lat,
            "lon": t_lon,
            "distance_km": None,
            "bearing_to_target_deg": None,
            "angle_deviation_deg": None,
            "probability_pct": 0,
            "eta_minutes": None,
            "risk_level": "low",
        }

        if active_lat is not None and active_lon is not None:
            dist_to_t = haversine_km(active_lat, active_lon, t_lat, t_lon)
            br_to_t = calculate_bearing(active_lat, active_lon, t_lat, t_lon)
            item["distance_km"] = round(dist_to_t, 1)
            item["bearing_to_target_deg"] = round(br_to_t, 1)
            eta_min = round((dist_to_t / max(speed_kmh, 30.0)) * 60.0)
            item["eta_minutes"] = max(eta_min, 1)

            if active_bearing is not None:
                dev = angle_diff_deg(active_bearing, br_to_t)
                item["angle_deviation_deg"] = round(dev, 1)
                # Формула вероятности угрозы: учитывает точность курса на цель (угловое отклонение) и дистанцию
                if dev <= 90.0:
                    angle_factor = math.cos(math.radians(dev)) ** 2  # от 1.0 при 0° до 0.0 при 90°
                    dist_factor = max(0.35, 1.0 - (dist_to_t / 450.0))
                    prob = int(round(min(98.0, max(5.0, angle_factor * dist_factor * 98.0))))
                else:
                    # Объект уходит в противоположную сторону
                    prob = max(2, int(round((180.0 - dev) / 90.0 * 12.0)))
                item["probability_pct"] = prob
            else:
                # Если вектор ещё не задан, оцениваем только по близости
                prob = max(10, min(75, int(round(100.0 / (1.0 + dist_to_t / 40.0)))))
                item["probability_pct"] = prob

            if item["probability_pct"] >= 65:
                item["risk_level"] = "high"
            elif item["probability_pct"] >= 35:
                item["risk_level"] = "medium"
            else:
                item["risk_level"] = "low"

        evaluated_targets.append(item)

    evaluated_targets.sort(key=lambda x: x["probability_pct"], reverse=True)

    return {
        "track": track_info,
        "targets_analysis": evaluated_targets,
    }

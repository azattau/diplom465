# -*- coding: utf-8 -*-
"""
Модуль трёх методов распознавания текста обращений 112:
1. Метод 1 (baseline)    — Базовый поиск точной подстроки в именительном падеже (алгоритм научного руководителя).
2. Метод 2 (morph_fuzzy) — Морфологический стемминг + синтаксические правила предлогов + нечёткий поиск по ФИАС.
3. Метод 3 (llm_hybrid)  — Большая языковая модель (Ollama LLM) + геокодер ФИАС (гибридный метод).
"""

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from geocoder import COMPASS_AZIMUTHS, EXTERNAL_DIRECTION_TARGETS, INVALID_PLACEHOLDERS, PVN_ALIASES, clean_settlement_name, extract_district_stem, normalize_text

DEFAULT_OLLAMA_URL = "http://localhost:11434"

STOPWORDS_RU = {
    "житель", "жители", "сообщает", "сообщил", "сообщила", "сообщают", "звонок",
    "видел", "видели", "вижу", "заметил", "заметили", "пролетел", "летит", "полетел",
    "сторону", "стороны", "направлению", "направляется", "около", "возле", "районе",
    "район", "района", "деревня", "деревни", "деревне", "деревней", "село", "села",
    "селе", "селом", "поселок", "поселка", "поселке", "город", "города", "городом",
    "хутор", "хутора", "беспилотник", "дрон", "бпла", "птица", "птицу", "объект",
    "воздушный", "небе", "низко", "высоко", "быстро", "звук", "мотора", "слышно",
    "север", "юг", "запад", "восток", "утро", "утром", "днем", "вечером", "ночью",
}

SYSTEM_PROMPT = """Ты — интеллектуальный модуль системы 112 / МЧС по Республике Башкортостан.
Твоя задача — извлечь из сообщения данные о замеченном воздушном объекте и вернуть СТРОГО валидный JSON.
Даже если сообщение состоит всего из одного слова (например, названия посёлка или села: "Шаран", "Михайловка Уфимский район"), обязательно запиши его в поле "settlement_seen" в именительном падеже.

Правила:
1. "settlement_seen" — название населённого пункта, ГДЕ заметили объект, строго в ИМЕНИТЕЛЬНОМ ПАДЕЖЕ без слов "село/деревня/поселок/город" (например: "Шаран", "Булатово", "Ишимбай", "Михайловка").
2. "settlement_type" — тип НП ("село", "деревня", "поселок", "город", "хутор" или "").
3. "district" — муниципальный район в именительном падеже, если он указан в тексте (например: "Шаранский", "Уфимский", "Абзелиловский", "Белебеевский" или "").
4. "direction_to" — населённый пункт, В СТОРОНУ КОТОРОГО летит объект, в именительном падеже (или "").
5. "direction_from" — населённый пункт, СО СТОРОНЫ КОТОРОГО летит объект, в именительном падеже (или "").
6. "direction_compass" — направление по сторонам света ("север", "северо-восток", "восток", "юго-восток", "юг", "юго-запад", "запад", "северо-запад" или "").
7. "event_time" — время из текста в формате "ЧЧ:ММ" (или "").
8. "object_details" — краткое описание объекта.

Верни строго JSON:
{
  "settlement_seen": "",
  "settlement_type": "",
  "district": "",
  "direction_to": "",
  "direction_from": "",
  "direction_compass": "",
  "event_time": "",
  "object_details": ""
}"""


def diagnose_ollama(ollama_url: str = DEFAULT_OLLAMA_URL) -> Dict:
    """
    Подробная диагностика локальной Ollama:
    проверяет, запущен ли сервис на порту 11434 и скачана ли хотя бы одна языковая модель.
    """
    base = ollama_url.rstrip("/")
    req = urllib.request.Request(f"{base}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            all_models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            gen_models = [m for m in all_models if "embed" not in m.lower()]
            models = gen_models if gen_models else all_models
            if len(models) == 0:
                return {
                    "online": True,
                    "ready": False,
                    "status_code": "no_models",
                    "models": [],
                    "message": "Программа Ollama запущена, но внутри неё ещё не скачана ни одна языковая модель! Выполните в командной строке: ollama pull qwen2.5:3b (или нажмите кнопку автозагрузки модели ниже).",
                }
            return {
                "online": True,
                "ready": True,
                "status_code": "ok",
                "models": models,
                "message": f"Ollama успешно подключена! Найдено моделей: {len(models)} ({', '.join(models[:3])}).",
            }
    except urllib.error.URLError as exc:
        return {
            "online": False,
            "ready": False,
            "status_code": "offline",
            "models": [],
            "message": f"Сервер Ollama не отвечает по адресу {base}. Убедитесь, что приложение Ollama запущено (значок ламы в трее возле часов) или выполните команду: ollama serve",
            "detail": str(exc),
        }
    except Exception as exc:
        return {
            "online": False,
            "ready": False,
            "status_code": "error",
            "models": [],
            "message": f"Ошибка при проверке Ollama: {exc}",
        }


def get_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> Tuple[bool, List[str]]:
    diag = diagnose_ollama(ollama_url)
    return diag["online"], diag["models"]


def pull_ollama_model(model_name: str = "qwen2.5:3b", ollama_url: str = DEFAULT_OLLAMA_URL) -> Tuple[bool, str]:
    """Скачивает модель в локальную Ollama через POST /api/pull."""
    url = f"{ollama_url.rstrip('/')}/api/pull"
    body = json.dumps({"name": model_name, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300.0) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            if res.get("status") == "success" or "success" in str(res).lower():
                return True, f"Модель {model_name} успешно скачана и готова к работе!"
            return True, f"Ответ Ollama: {res}"
    except Exception as exc:
        return False, f"Не удалось автоматически скачать модель ({exc}). Откройте терминал и выполните: ollama pull {model_name}"


# =====================================================================
# МЕТОД 1: Базовый метод (алгоритм научного руководителя — точная подстрока в им. падеже)
# =====================================================================
def method_1_baseline(message_text: str, geocoder) -> Dict:
    """
    Воспроизводит базовый алгоритм из исходного ноутбука generate_dataset.ipynb руководителя:
    ищет прямое вхождение названия НП в именительном падеже (без пробелов) в текст сообщения,
    без разбора косвенных падежей, без определения района и без построения вектора.
    """
    phrase_lower = (message_text or "").strip().lower().replace("ё", "е")
    words_set = set(re.findall(r"[а-я0-9\-]+", phrase_lower))

    time_match = re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", phrase_lower)
    event_time = f"{int(time_match.group(1)):02d}:{time_match.group(2)}" if time_match else datetime.now().strftime("%H:%M")

    found_name = ""
    best_len = 0
    for rec in geocoder.records:
        city_nospace = rec["nospace_name"]
        if len(city_nospace) < 3:
            continue
        # Проверяем точное совпадение слова в именительном падеже (чтобы не ловить случайные 3 буквы внутри чужих слов)
        if city_nospace in words_set and len(city_nospace) > best_len:
            found_name = rec["clean_name"].title()
            best_len = len(city_nospace)

    return {
        "settlement_seen": found_name,
        "settlement_type": "",
        "district": "",
        "direction_to": "",
        "direction_from": "",
        "direction_compass": "",
        "event_time": event_time,
        "object_details": "Метод 1 (базовый поиск точного совпадения в им. падеже без вектора)",
    }


# =====================================================================
# МЕТОД 2: Морфологический + Нечёткий поиск (стемминг + правила предлогов + опечатки)
# =====================================================================
def _bigram_similarity(s1: str, s2: str) -> float:
    """Вычисляет коэффициент Сёренсена-Дайса по символьным биграммам для устойчивости к опечаткам."""
    if s1 == s2:
        return 1.0
    if len(s1) < 2 or len(s2) < 2:
        return 0.0
    b1 = {s1[i : i + 2] for i in range(len(s1) - 1)}
    b2 = {s2[i : i + 2] for i in range(len(s2) - 1)}
    return (2.0 * len(b1 & b2)) / (len(b1) + len(b2))


def method_2_morph_fuzzy(message_text: str, geocoder) -> Dict:
    """
    Быстрый алгоритмический анализатор:
    - Выделяет роли топонимов по предложным конструкциям («над...», «в сторону...», «со стороны...»);
    - Приводит косвенные падежи к именительному через стемминг основ справочника settlments.csv;
    - Определяет муниципальный район для одноимённых деревень/посёлков;
    - Исправляет опечатки за счёт сравнения символьных биграмм.
    """
    text_clean = (message_text or "").strip()
    text_lower = text_clean.lower().replace("ё", "е")
    text_lower = re.sub(r"([а-я])(\d)", r"\1 \2", text_lower)
    text_lower = re.sub(r"(\d)([а-я])", r"\1 \2", text_lower)
    text_lower = re.sub(r"республик[а-я]*\s+башкортостан", "рб", text_lower)
    text_lower = re.sub(r"\bн\.?\s*п\.?\s+", "село ", text_lower)
    text_lower = re.sub(r"\bв\s+с[тчь]орону\b", "в сторону", text_lower)
    text_lower = re.sub(r"\bв\s+направлении\b", "в сторону", text_lower)

    time_match = re.search(r"(?:\b|время\s*)([01]?\d|2[0-3])[:.\-,]([0-5]\d)\b", text_lower)
    event_time = f"{int(time_match.group(1)):02d}:{time_match.group(2)}" if time_match else datetime.now().strftime("%H:%M")

    # Извлечение муниципального района
    district = ""
    dist_match = re.search(
        r"([а-я\-]{3,}?)(?:ского|цкого|ого|его|ском|цком|ом|ем|ский|цкий|ый|ий)\s+(?:муниципального\s+|муниципальный\s+)?(?:район|р-н)",
        text_lower,
    )
    if dist_match:
        stem = extract_district_stem(dist_match.group(1))
        district = geocoder.district_stems.get(stem, dist_match.group(1).capitalize() + "ский")
    else:
        words_in_msg = re.findall(r"[а-я\-]{4,}", text_lower)
        for w in words_in_msg:
            if re.search(r"(ский|цкий|ского|цкого|ском|цком|скому|цкому)$", w):
                st = extract_district_stem(w)
                if st in geocoder.district_stems:
                    district = geocoder.district_stems[st]
                    break

    # Направление по сторонам света
    direction_compass = ""
    for comp in ["северо-восток", "северо-запад", "юго-восток", "юго-запад", "север", "восток", "юг", "запад"]:
        if re.search(r"\b(?:на|курс|в\s+сторону)\s+" + comp, text_lower) or re.search(r"\b" + comp + r"(?:ном|нее)\b", text_lower):
            direction_compass = comp
            break

    settlement_type = ""
    for pat, t_norm in [
        (r"\b(деревн[а-я]*|д\.|дер\.)\b", "деревня"),
        (r"\b(сел[а-я]*|с\.)\b", "село"),
        (r"\b(посел[а-я]*|п\.|пос\.|рп|пгт)\b", "поселок"),
        (r"\b(город[а-я]*|г\.)\b", "город"),
        (r"\b(хутор[а-я]*|х\.)\b", "хутор"),
    ]:
        if re.search(pat, text_lower):
            settlement_type = t_norm
            break

    def match_toponym_in_fragment(fragment: str, allow_external: bool = False) -> Tuple[str, str]:
        frag_norm = normalize_text(fragment)
        if not frag_norm:
            return "", ""

        if allow_external:
            for ext_key in EXTERNAL_DIRECTION_TARGETS:
                if re.search(r"\b" + re.escape(ext_key), frag_norm):
                    return ext_key, "ориентир"
        for al_key, (al_target, _) in PVN_ALIASES.items():
            if re.search(r"\b" + re.escape(al_key) + r"\b", frag_norm):
                return al_target.title(), "село"


        direct_clean, direct_type = clean_settlement_name(frag_norm)
        if direct_clean:
            for rec in geocoder.records:
                if rec["clean_name"] == direct_clean or rec["nospace_name"] == direct_clean.replace(" ", ""):
                    return rec["clean_name"].title(), (direct_type or rec["type"])

        frag_words = [
            w for w in re.findall(r"[а-я]{3,}", frag_norm)
            if w not in STOPWORDS_RU and not re.search(r"(ского|цкого|ском|цком|ский|цкий)$", w)
        ]
        if not frag_words:
            frag_words = [w for w in re.findall(r"[а-я]{3,}", frag_norm) if w not in STOPWORDS_RU]

        best_match = ("", "", 0.0)
        for rec in geocoder.records:
            cname = rec["clean_name"]
            if len(cname) < 3:
                continue
            cwords = [x for x in re.split(r"[\s\-]+", cname) if x]

            if len(cwords) == 1:
                w = cwords[0]
                if w in frag_words:
                    score = len(w) * 4.0 + (25.0 if rec["type"] == "город" else (10.0 if rec["type"] == "село" else 0.0)) + 30.0
                    if score > best_match[2]:
                        best_match = (cname.title(), rec["type"], score)
                    continue

                if len(w) <= 4:
                    stem = w[:-1] if w[-1] in "аяоеиыуьюй" else w
                    pattern = r"\b" + re.escape(stem) + r"(?:а|я|у|ю|е|ой|ей|ою|ею|ы|и|ам|ям|ами|ями|ах|ях|ом|ем)?\b"
                    if re.search(pattern, frag_norm):
                        score = len(w) * 3.0 + (25.0 if rec["type"] == "город" else 0.0)
                        if score > best_match[2]:
                            best_match = (cname.title(), rec["type"], score)
                else:
                    stem = (
                        w[:-2]
                        if w.endswith(("ово", "ево", "ино", "ыно", "ский", "цкий", "ая", "ое", "ий", "ый"))
                        else (w[:-1] if w[-1] in "аяоеиыуьюй" else w)
                    )
                    if len(stem) >= 3:
                        for fw in frag_words:
                            if fw.startswith(stem) and abs(len(fw) - len(stem)) <= 3:
                                diff_penalty = abs(len(fw) - len(w)) * 4.0
                                city_bonus = 25.0 if rec["type"] == "город" else 0.0
                                score = len(stem) * 3.0 + city_bonus - diff_penalty
                                if score > best_match[2]:
                                    best_match = (cname.title(), rec["type"], score)
                            elif len(fw) >= 5 and len(w) >= 5 and abs(len(fw) - len(w)) <= 2:
                                # Нечёткое сравнение для защиты от опечаток (например, "Шоран", "Стерлетомак")
                                sim = _bigram_similarity(fw[: len(stem)], stem)
                                if sim >= 0.78:
                                    score = len(stem) * 2.2 * sim + (15.0 if rec["type"] == "город" else 0.0)
                                    if score > best_match[2]:
                                        best_match = (cname.title(), rec["type"], score)
            else:
                # Каждая часть составного названия должна иметь минимум 3 буквы (или точное совпадение слова),
                # и минимум одна часть должна быть >= 4 букв
                if not any(len(cw) >= 4 for cw in cwords):
                    if cname in frag_norm.split():
                        score = len(cname) * 4.0
                        if score > best_match[2]:
                            best_match = (cname.title(), rec["type"], score)
                    continue
                all_tokens_in_frag = set(re.findall(r"[а-я0-9]+", frag_norm))
                matched_all = True
                for cw in cwords:
                    if len(cw) < 3:
                        if cw not in all_tokens_in_frag:
                            matched_all = False
                            break
                    else:
                        st = cw[:-2] if len(cw) > 5 else (cw[:-1] if cw[-1] in "аяоеиыуьюй" else cw)
                        if not any(fw.startswith(st) and abs(len(fw) - len(cw)) <= 3 for fw in frag_words):
                            matched_all = False
                            break
                if matched_all:
                    score = len(cname) * 5.0 + 40.0
                    if score > best_match[2]:
                        best_match = (cname.title(), rec["type"], score)

        return best_match[0], best_match[1]

    direction_to = ""
    direction_from = ""
    rest_text = text_lower

    dir_to_match = re.search(
        r"(?:в\s+сторону(?:\s+в)?|по\s+направлению\s+к|направляется\s+к|летит\s+к|полетел\s+к|полетел\s+на|курс\s+на|летит\s+на)\s+((?:(?:г|с|д|п|х|ст)\.?\s+)?[а-я\-]{2,25}(?:\s+[а-я\-]{3,20})?)",
        text_lower,
    )
    if dir_to_match:
        dir_frag = dir_to_match.group(1).strip()
        first_w = re.sub(r"^(?:г|с|д|п|х|ст)\.?\s+", "", dir_frag).split()[0]
        if first_w not in COMPASS_AZIMUTHS and dir_frag not in COMPASS_AZIMUTHS:
            direction_to, _ = match_toponym_in_fragment(first_w, allow_external=True)
            matched_span = first_w
            if not direction_to:
                direction_to, _ = match_toponym_in_fragment(dir_frag, allow_external=True)
                matched_span = dir_frag
            if direction_to:
                idx_span = dir_to_match.group(0).find(matched_span)
                prefix_part = dir_to_match.group(0)[: idx_span + len(matched_span)] if idx_span != -1 else dir_to_match.group(0)
                rest_text = text_lower.replace(prefix_part, " ", 1)

    dir_from_match = re.search(
        r"(?:со\s+стороны|от\s+города|от\s+села|от\s+деревни|от\s+поселка|из\s+района)\s+([а-я\-\s.]{3,30}?)(?:[,.;]|$|\s+в\s+|\s+над\s+|\s+по\s+)",
        rest_text,
    )
    if dir_from_match:
        from_frag = dir_from_match.group(1).strip()
        direction_from, _ = match_toponym_in_fragment(from_frag, allow_external=True)
        if direction_from:
            rest_text = rest_text.replace(dir_from_match.group(0), " ")

    settlement_seen, detected_t = match_toponym_in_fragment(rest_text)
    if not settlement_type and detected_t:
        settlement_type = detected_t

    return {
        "settlement_seen": settlement_seen,
        "settlement_type": settlement_type,
        "district": district,
        "direction_to": direction_to,
        "direction_from": direction_from,
        "direction_compass": direction_compass,
        "event_time": event_time,
        "object_details": "Воздушный объект (Метод 2: Морфология + нечёткий поиск)",
    }


# Для обратной совместимости с ноутбуком
morph_parse_message = method_2_morph_fuzzy


# =====================================================================
# МЕТОД 3: Нейросетевой / Гибридный (Локальная LLM Ollama + верификация ФИАС)
# =====================================================================
def call_ollama_json(
    message_text: str,
    morph_hint: Dict,
    model_name: Optional[str] = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Tuple[bool, Dict, str]:
    diag = diagnose_ollama(ollama_url)
    if not diag["ready"]:
        return False, {"reason": diag["message"]}, diag["status_code"]

    models = diag["models"]
    chosen_model = model_name if (model_name and model_name in models) else models[0]

    hint_str = (
        f"Подсказка морфологического словаря settlments.csv: "
        f"кандидат НП='{morph_hint.get('settlement_seen', '')}', "
        f"район='{morph_hint.get('district', '')}', "
        f"направление='{morph_hint.get('direction_to', '') or morph_hint.get('direction_compass', '')}'."
    )

    url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Сообщение: {message_text}\n{hint_str}"},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=45.0) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result.get("message", {}).get("content", "{}")
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return True, parsed, chosen_model
    except Exception as exc:
        return False, {"error": str(exc)}, chosen_model

    return False, {}, chosen_model


def analyze_by_method(
    message_text: str,
    geocoder,
    method: str = "method_3_hybrid",
    preferred_model: Optional[str] = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> Tuple[Dict, str, float]:
    """
    Запускает выбранный метод распознавания текста (1, 2 или 3) и замеряет время работы в миллисекундах.
    Возвращает: (извлеченный_json, название_метода, время_в_мс).
    """
    t0 = time.perf_counter()

    if method in ("method_1_baseline", "baseline"):
        res = method_1_baseline(message_text, geocoder)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res, "Метод 1: Базовый (точная подстрока в им. падеже)", elapsed_ms

    if method in ("method_2_morph", "morph_only", "morph_fuzzy"):
        res = method_2_morph_fuzzy(message_text, geocoder)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res, "Метод 2: Морфологический + Нечёткий поиск (ФИАС)", elapsed_ms

    # Метод 3: Гибридный (LLM Ollama + Морфология ФИАС)
    morph_result = method_2_morph_fuzzy(message_text, geocoder)
    used_ollama, llm_raw, model_or_status = call_ollama_json(
        message_text=message_text,
        morph_hint=morph_result,
        model_name=preferred_model,
        ollama_url=ollama_url,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not used_ollama:
        return (
            morph_result,
            "Метод 3: Гибрид (сработал морфологический контур, т.к. Ollama не подключена)",
            elapsed_ms,
        )

    llm_seen = (
        llm_raw.get("settlement_seen")
        or llm_raw.get("settlement")
        or llm_raw.get("city")
        or llm_raw.get("location")
        or ""
    ).strip()
    if llm_seen.lower() in INVALID_PLACEHOLDERS:
        llm_seen = ""

    llm_cand = geocoder.find_settlement(llm_seen) if llm_seen else None
    final_seen = llm_seen if llm_cand else morph_result["settlement_seen"]

    llm_dir_to = (llm_raw.get("direction_to") or "").strip()
    if llm_dir_to.lower() in INVALID_PLACEHOLDERS or not geocoder.find_settlement(llm_dir_to):
        llm_dir_to = morph_result["direction_to"]

    llm_dist = (llm_raw.get("district") or "").strip()
    if llm_dist.lower() in INVALID_PLACEHOLDERS or not extract_district_stem(llm_dist):
        llm_dist = morph_result["district"]

    merged = {
        "settlement_seen": final_seen,
        "settlement_type": (llm_raw.get("settlement_type") or morph_result["settlement_type"] or "").strip(),
        "district": llm_dist,
        "direction_to": llm_dir_to,
        "direction_from": (llm_raw.get("direction_from") or morph_result["direction_from"] or "").strip(),
        "direction_compass": (llm_raw.get("direction_compass") or morph_result["direction_compass"] or "").strip(),
        "event_time": (llm_raw.get("event_time") or morph_result["event_time"] or "").strip(),
        "object_details": (llm_raw.get("object_details") or "Воздушный объект (Метод 3: LLM + ФИАС)").strip(),
    }

    return merged, f"Метод 3: Гибрид Ollama ({model_or_status}) + ФИАС", elapsed_ms


def compare_all_three_methods(
    message_text: str,
    geocoder,
    preferred_model: Optional[str] = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> List[Dict]:
    """
    Прогоняет одно сообщение через все 3 метода распознавания и возвращает сравнительную таблицу
    (для демонстрации научному руководителю).
    """
    comparison = []
    for m_code in ["method_1_baseline", "method_2_morph", "method_3_hybrid"]:
        extracted, label, ms = analyze_by_method(
            message_text=message_text,
            geocoder=geocoder,
            method=m_code,
            preferred_model=preferred_model,
            ollama_url=ollama_url,
        )
        geo = geocoder.resolve_observation(extracted)
        settlement = geo.get("settlement") or {}
        vector = geo.get("vector") or {}
        comparison.append({
            "method_code": m_code,
            "method_label": label,
            "elapsed_ms": ms,
            "found": geo.get("found", False),
            "extracted_nominative": extracted.get("settlement_seen") or "—",
            "matched_full": settlement.get("full_name") or "Не найден",
            "district": settlement.get("district") or "—",
            "vector_desc": vector.get("description") or "Вектор не определён",
            "homonyms_count": geo.get("total_homonyms", 0),
        })
    return comparison


# Алиас для совместимости
def hybrid_analyze_message(message_text: str, geocoder, preferred_model: Optional[str] = None, ollama_url: str = DEFAULT_OLLAMA_URL, mode: str = "method_3_hybrid") -> Tuple[Dict, str]:
    ext, label, _ = analyze_by_method(message_text, geocoder, method=mode, preferred_model=preferred_model, ollama_url=ollama_url)
    return ext, label

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Локальный веб-сервер системы мониторинга воздушных объектов:
- 3 метода распознавания текста (Базовый, Морфология+Нечёткий, Гибрид с LLM Ollama);
- Построение единого трека пролёта, расчёт скорости и прогноз положения (+15 / +30 мин);
- Добавление оператором охраняемых объектов (целей) на карту и расчёт вероятности угрозы (%).
"""

import json
import os
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from geocoder import SettlementGeocoder, compute_track_and_threats
from llm_client import (
    DEFAULT_OLLAMA_URL,
    analyze_by_method,
    compare_all_three_methods,
    diagnose_ollama,
    pull_ollama_model,
)

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "settlments.csv")
OBS_JSON_PATH = os.path.join(BASE_DIR, "observations.json")
TARGETS_JSON_PATH = os.path.join(BASE_DIR, "targets.json")
COORDS_TXT_PATH = os.path.join(BASE_DIR, "coords.txt")

print("Загрузка справочника населённых пунктов Республики Башкортостан (settlments.csv)...")
GEOCODER = SettlementGeocoder(CSV_PATH)
print(f"Загружено населённых пунктов: {len(GEOCODER.records)}, районов/округов: {len(GEOCODER.districts)}")


def load_observations():
    if os.path.exists(OBS_JSON_PATH):
        try:
            with open(OBS_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_observations(items):
    with open(OBS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    with open(COORDS_TXT_PATH, "w", encoding="utf-8") as f:
        for item in items:
            clean_msg = item.get("raw_text", "").replace("\n", " ").replace(",", ";")
            f.write(f"{item['lat']},{item['lon']},[{item['time']}] {clean_msg}\n")


def load_targets():
    if os.path.exists(TARGETS_JSON_PATH):
        try:
            with open(TARGETS_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_targets(items):
    with open(TARGETS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def build_state_payload(extra=None):
    obs = load_observations()
    targets = load_targets()
    analytics = compute_track_and_threats(obs, targets)
    res = {
        "ok": True,
        "observations": obs,
        "targets": targets,
        "analytics": analytics,
    }
    if extra:
        res.update(extra)
    return res


class OperatorRequestHandler(SimpleHTTPRequestHandler):
    """Обработчик HTTP-запросов веб-приложения оператора 112."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def _send_json(self, status_code: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            diag = diagnose_ollama(DEFAULT_OLLAMA_URL)
            self._send_json(200, {
                "backend": "python",
                "ollama_online": diag["online"],
                "ollama_ready": diag["ready"],
                "ollama_status_code": diag["status_code"],
                "ollama_message": diag["message"],
                "ollama_url": DEFAULT_OLLAMA_URL,
                "models": diag["models"],
                "settlements_count": len(GEOCODER.records),
                "districts": GEOCODER.districts,
            })
            return

        if parsed.path == "/api/observations":
            self._send_json(200, build_state_payload())
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            body = json.loads(raw_body)
        except Exception:
            body = {}

        if parsed.path == "/api/load_pvn_sample":
            limit = int(body.get("limit") or 12)
            offset = int(body.get("offset") or 0)
            rep_path = os.path.join(BASE_DIR, "report.txt")
            observations = load_observations()
            added_count = 0
            if os.path.exists(rep_path):
                with open(rep_path, "r", encoding="utf-8") as rf:
                    raw_lines = [l.strip() for l in rf if len(l.strip()) > 18 and "Оренбургской области" not in l]
                for idx, line_text in enumerate(raw_lines[offset:]):
                    if added_count >= limit:
                        break
                    ext_json, eng_lbl, el_ms = analyze_by_method(line_text, GEOCODER, method="method_2_morph")
                    if not ext_json.get("settlement_seen"):
                        continue
                    geo_res = GEOCODER.resolve_observation(ext_json)
                    if not geo_res.get("found"):
                        continue
                    s = geo_res["settlement"]
                    if str(s["id"]).startswith("ext_"):
                        continue
                    observations.append({
                        "id": int(datetime.now().timestamp() * 1000) + idx,
                        "time": ext_json.get("event_time") or "09:30",
                        "created_at": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                        "raw_text": line_text,
                        "settlement_id": s["id"],
                        "lat": s["lat"],
                        "lon": s["lon"],
                        "settlement_full": s["full_name"],
                        "settlement_clean": s["clean_name"].title(),
                        "settlement_type": s["type"],
                        "district": s["district"],
                        "fias_id": s["fias_id"],
                        "object_details": "Из сводки ПВН (report.txt)",
                        "vector": geo_res["vector"],
                        "llm_extracted": ext_json,
                        "engine": f"{eng_lbl} ({el_ms} мс)",
                        "is_ambiguous": geo_res.get("is_ambiguous", False),
                        "total_homonyms": geo_res.get("total_homonyms", 1),
                        "alternatives": geo_res.get("alternatives", []),
                    })
                    added_count += 1
                save_observations(observations)
            self._send_json(200, build_state_payload({"added_count": added_count}))
            return

        if parsed.path == "/api/clear":
            save_observations([])
            self._send_json(200, build_state_payload())
            return

        if parsed.path == "/api/add_target":
            name = (body.get("name") or "Охраняемый объект").strip()
            settlement_query = (body.get("settlement") or "").strip()
            lat = body.get("lat")
            lon = body.get("lon")

            if settlement_query and (lat is None or lon is None):
                rec = GEOCODER.find_settlement(settlement_query)
                if not rec:
                    self._send_json(200, {"ok": False, "error": f"Населённый пункт «{settlement_query}» не найден в справочнике."})
                    return
                lat, lon = rec["lat"], rec["lon"]
                if name == "Охраняемый объект":
                    name = f"Объект ({rec['full_name']})"

            try:
                lat_f = round(float(lat), 6)
                lon_f = round(float(lon), 6)
            except (TypeError, ValueError):
                self._send_json(400, {"ok": False, "error": "Укажите корректные координаты или название населённого пункта."})
                return

            targets = load_targets()
            new_target = {
                "id": int(datetime.now().timestamp() * 1000),
                "name": name,
                "lat": lat_f,
                "lon": lon_f,
            }
            targets.append(new_target)
            save_targets(targets)
            self._send_json(200, build_state_payload({"added_target": new_target}))
            return

        if parsed.path == "/api/delete_target":
            target_id = body.get("id")
            if target_id == "all":
                save_targets([])
            else:
                targets = [t for t in load_targets() if str(t.get("id")) != str(target_id)]
                save_targets(targets)
            self._send_json(200, build_state_payload())
            return

        if parsed.path == "/api/preset_targets":
            # 3 стратегических узла РБ для быстрой демонстрации
            presets = [
                {"id": 101, "name": "Цель №1 (Уфимский узел)", "lat": 54.735598, "lon": 55.957605},
                {"id": 102, "name": "Цель №2 (Салават / Стерлитамак)", "lat": 53.362823, "lon": 55.915437},
                {"id": 103, "name": "Цель №3 (Северо-Запад / Нефтекамск)", "lat": 56.088375, "lon": 54.248241},
            ]
            save_targets(presets)
            self._send_json(200, build_state_payload())
            return

        if parsed.path == "/api/ollama_pull":
            model_to_pull = (body.get("model") or "qwen2.5:3b").strip()
            ok, msg = pull_ollama_model(model_to_pull, DEFAULT_OLLAMA_URL)
            diag = diagnose_ollama(DEFAULT_OLLAMA_URL)
            self._send_json(200, {
                "ok": ok,
                "message": msg,
                "models": diag["models"],
                "ollama_ready": diag["ready"],
            })
            return

        if parsed.path == "/api/compare":
            message_text = (body.get("text") or "").strip()
            preferred_model = (body.get("model") or "").strip()
            ollama_url = (body.get("ollama_url") or DEFAULT_OLLAMA_URL).strip()
            if not message_text:
                self._send_json(400, {"ok": False, "error": "Введите текст сообщения для сравнения методов."})
                return
            comp = compare_all_three_methods(
                message_text=message_text,
                geocoder=GEOCODER,
                preferred_model=preferred_model or None,
                ollama_url=ollama_url,
            )
            self._send_json(200, {"ok": True, "comparison": comp})
            return

        if parsed.path == "/api/switch_district":
            obs_id = body.get("obs_id")
            settlement_id = str(body.get("settlement_id") or "")
            observations = load_observations()
            updated_obs = None

            for obs in observations:
                if str(obs.get("id")) == str(obs_id):
                    extracted = obs.get("llm_extracted") or {}
                    geo_res = GEOCODER.resolve_observation(extracted, forced_settlement_id=settlement_id)
                    if geo_res.get("found"):
                        s = geo_res["settlement"]
                        obs["lat"] = s["lat"]
                        obs["lon"] = s["lon"]
                        obs["settlement_id"] = s["id"]
                        obs["settlement_full"] = s["full_name"]
                        obs["settlement_clean"] = s["clean_name"].title()
                        obs["settlement_type"] = s["type"]
                        obs["district"] = s["district"]
                        obs["fias_id"] = s["fias_id"]
                        obs["vector"] = geo_res["vector"]
                        obs["is_ambiguous"] = False
                        updated_obs = obs
                    break

            if updated_obs:
                save_observations(observations)
                self._send_json(200, build_state_payload({"observation": updated_obs}))
            else:
                self._send_json(400, {"ok": False, "error": "Не удалось переключить район."})
            return

        if parsed.path == "/api/analyze":
            message_text = (body.get("text") or "").strip()
            preferred_model = (body.get("model") or "").strip()
            custom_time = (body.get("time") or "").strip()
            district_override = (body.get("district") or "").strip()
            forced_settlement_id = str(body.get("forced_settlement_id") or "").strip()
            method = (body.get("mode") or body.get("method") or "method_3_hybrid").strip()
            ollama_url = (body.get("ollama_url") or DEFAULT_OLLAMA_URL).strip()

            if not message_text:
                self._send_json(400, {"ok": False, "error": "Введите текст сообщения."})
                return

            extracted_json, engine_label, elapsed_ms = analyze_by_method(
                message_text=message_text,
                geocoder=GEOCODER,
                method=method,
                preferred_model=preferred_model or None,
                ollama_url=ollama_url,
            )

            if district_override:
                extracted_json["district"] = district_override

            geo_result = GEOCODER.resolve_observation(extracted_json, forced_settlement_id=forced_settlement_id)
            if not geo_result.get("found"):
                err_msg = geo_result.get("error", "Населённый пункт не найден в справочнике.")
                if method == "method_1_baseline":
                    err_msg = (
                        "Метод 1 (базовый алгоритм руководителя) не смог распознать НП, так как название стоит "
                        "в косвенном падеже или состоит из нескольких слов. Переключите на Метод 2 или Метод 3!"
                    )
                self._send_json(200, {
                    "ok": False,
                    "error": err_msg,
                    "llm_extracted": extracted_json,
                    "engine": f"{engine_label} ({elapsed_ms} мс)",
                })
                return

            settlement = geo_result["settlement"]
            vector = geo_result["vector"]

            now_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            event_time = custom_time or extracted_json.get("event_time") or datetime.now().strftime("%H:%M")

            observation = {
                "id": int(datetime.now().timestamp() * 1000),
                "time": event_time,
                "created_at": now_str,
                "raw_text": message_text,
                "settlement_id": settlement["id"],
                "lat": settlement["lat"],
                "lon": settlement["lon"],
                "settlement_full": settlement["full_name"],
                "settlement_clean": settlement["clean_name"].title(),
                "settlement_type": settlement["type"],
                "district": settlement["district"],
                "fias_id": settlement["fias_id"],
                "object_details": extracted_json.get("object_details") or "Воздушный объект",
                "vector": vector,
                "llm_extracted": extracted_json,
                "engine": f"{engine_label} ({elapsed_ms} мс)",
                "elapsed_ms": elapsed_ms,
                "is_ambiguous": geo_result.get("is_ambiguous", False),
                "total_homonyms": geo_result.get("total_homonyms", 1),
                "alternatives": geo_result.get("alternatives", []),
            }

            observations = load_observations()
            observations.append(observation)
            save_observations(observations)

            self._send_json(200, build_state_payload({"observation": observation}))
            return

        self._send_json(404, {"ok": False, "error": "Неизвестный эндпоинт"})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), OperatorRequestHandler)
    print("=" * 72)
    print(f"Система мониторинга воздушных объектов (Треки + Цели оператора + 3 метода) запущена!")
    print(f"Откройте в браузере: http://localhost:{PORT}")
    print("Для остановки сервера нажмите Ctrl+C")
    print("=" * 72)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")

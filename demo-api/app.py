"""
Mabna public demo API — Wind pressure generator.

Self-contained, standalone service for the landing-site demos. It deliberately
does NOT import the real Structure_Engine; it re-implements a *simplified* subset
of the ASCE 7-22 MWFRS directional procedure using the standard public formulas.
No database, no auth, no secrets — safe to expose publicly.

Abuse controls:
  - inputs are strictly validated numerics (no injection surface),
  - a per-IP flood guard (min interval between calls),
  - a hard free-run cap per IP (after N runs, the API returns 403 and the page
    asks the visitor to sign up for the full app).
"""
from __future__ import annotations

import math
import time
import threading
from collections import defaultdict

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Mabna Demo API", version="1.0.0", docs_url=None, redoc_url=None)

# Only the landing site may call this. Add localhost for local testing.
ALLOWED_ORIGINS = [
    "https://mabnas.com",
    "https://www.mabnas.com",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
    max_age=600,
)

# ── Abuse controls ────────────────────────────────────────────────────────
WIND_FREE_RUNS = 3          # wind tool: a real calc, strict cap → sign-up gate
FRAME_FREE_RUNS = 15        # frame tool: just geometry drawing, more generous
WINDOW_SECONDS = 24 * 3600  # rolling window the cap applies over
MIN_INTERVAL = 0.3          # min seconds between calls from one IP (flood guard)

_lock = threading.Lock()
_hits: dict[str, list[float]] = defaultdict(list)  # "tool:ip" -> run timestamps
_last: dict[str, float] = {}                        # "tool:ip" -> last request time


def _client_ip(request: Request) -> str:
    # Behind Caddy, the real IP is the first hop of X-Forwarded-For.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_and_count(ip: str, tool: str, limit: int) -> int:
    """Flood-guard + per-tool free-run cap. Returns the run number (1-based) on
    success, raises HTTPException otherwise. Call once inputs are valid."""
    now = time.time()
    key = f"{tool}:{ip}"
    with _lock:
        last = _last.get(key, 0.0)
        if now - last < MIN_INTERVAL:
            raise HTTPException(429, detail={"error": "too_fast", "message": "Slow down a moment and try again."})
        _last[key] = now

        hits = [t for t in _hits[key] if now - t < WINDOW_SECONDS]
        if len(hits) >= limit:
            _hits[key] = hits
            raise HTTPException(
                403,
                detail={
                    "error": "limit_reached",
                    "message": f"You've used your {limit} free demo runs. Sign up to use the full app without limits.",
                    "sign_up_url": "https://app.mabnas.com",
                },
            )
        hits.append(now)
        _hits[key] = hits

        # opportunistic cleanup so the dicts don't grow forever
        if len(_hits) > 5000:
            for k in list(_hits):
                _hits[k] = [t for t in _hits[k] if now - t < WINDOW_SECONDS]
                if not _hits[k]:
                    del _hits[k]
                    _last.pop(k, None)

        return len(hits)


# ── ASCE 7-22 exposure constants (Table 26.11-1), metric zg ───────────────
_EXPOSURE = {
    "B": {"alpha": 7.0, "zg": 365.76, "zmin": 9.14},
    "C": {"alpha": 9.5, "zg": 274.32, "zmin": 4.57},
    "D": {"alpha": 11.5, "zg": 213.36, "zmin": 2.13},
}
_KD = 0.85   # directionality factor, buildings (MWFRS)
_KZT = 1.0   # topographic factor (flat terrain)
_KE = 1.0    # ground elevation factor (sea level)
_G = 0.85    # gust-effect factor (rigid building)


class WindIn(BaseModel):
    units: str = Field("si", description="'si' (m/s, kPa) or 'imperial' (mph, psf)")
    wind_speed: float = Field(..., gt=0, le=400, description="Basic wind speed V")
    exposure: str = Field("C")
    mean_roof_height: float = Field(..., gt=0, le=1000, description="h")
    length_parallel: float = Field(..., gt=0, le=2000, description="L (parallel to wind)")
    width_normal: float = Field(..., gt=0, le=2000, description="B (normal to wind)")
    enclosure: str = Field("enclosed")

    @field_validator("units")
    @classmethod
    def _u(cls, v):
        v = v.lower()
        if v not in ("si", "imperial"):
            raise ValueError("units must be 'si' or 'imperial'")
        return v

    @field_validator("exposure")
    @classmethod
    def _e(cls, v):
        v = v.upper()
        if v not in _EXPOSURE:
            raise ValueError("exposure must be B, C or D")
        return v

    @field_validator("enclosure")
    @classmethod
    def _c(cls, v):
        v = v.lower()
        if v not in ("enclosed", "partial"):
            raise ValueError("enclosure must be 'enclosed' or 'partial'")
        return v


def _kz(z: float, exp: dict) -> float:
    z_eval = max(z, exp["zmin"])
    return 2.01 * (z_eval / exp["zg"]) ** (2.0 / exp["alpha"])


def _leeward_cp(l_over_b: float) -> float:
    # ASCE 7-22 Fig 27.3-1 leeward wall Cp, interpolated on L/B.
    if l_over_b <= 1.0:
        return -0.5
    if l_over_b >= 4.0:
        return -0.2
    if l_over_b <= 2.0:
        return -0.5 + (l_over_b - 1.0) * ((-0.3) - (-0.5)) / (2.0 - 1.0)
    return -0.3 + (l_over_b - 2.0) * ((-0.2) - (-0.3)) / (4.0 - 2.0)


def compute_wind(inp: WindIn) -> dict:
    exp = _EXPOSURE[inp.exposure]
    si = inp.units == "si"
    coef = 0.613 if si else 0.00256   # qz constant
    punit = "kPa" if si else "psf"
    vunit = "m/s" if si else "mph"
    lunit = "m" if si else "ft"

    kz = _kz(inp.mean_roof_height, exp)
    qh = coef * kz * _KZT * _KD * _KE * inp.wind_speed ** 2   # N/m² (si) or psf
    gcpi = 0.18 if inp.enclosure == "enclosed" else 0.55

    l_over_b = inp.length_parallel / inp.width_normal
    cp_wind, cp_lee, cp_side = 0.80, _leeward_cp(l_over_b), -0.70

    def to_disp(p_pa_or_psf: float) -> float:
        # qh is N/m² in SI → convert to kPa; imperial already psf.
        return round((p_pa_or_psf / 1000.0) if si else p_pa_or_psf, 3)

    p_wind = qh * _G * cp_wind
    p_lee = qh * _G * cp_lee
    p_side = qh * _G * cp_side
    p_int = qh * gcpi

    return {
        "inputs": inp.model_dump(),
        "derived": {
            "Kz": round(kz, 3), "Kd": _KD, "Kzt": _KZT, "Ke": _KE, "G": _G,
            "GCpi": gcpi, "L_over_B": round(l_over_b, 2),
            "qh": to_disp(qh), "qh_unit": punit,
        },
        "surfaces": [
            {"name": "Windward wall", "Cp": round(cp_wind, 2), "pressure": to_disp(p_wind), "kind": "pressure"},
            {"name": "Leeward wall", "Cp": round(cp_lee, 2), "pressure": to_disp(p_lee), "kind": "suction"},
            {"name": "Side walls", "Cp": round(cp_side, 2), "pressure": to_disp(p_side), "kind": "suction"},
        ],
        "internal_pressure": {"value": to_disp(p_int), "unit": punit, "note": "±GCpi·qh (add/subtract per ASCE 7 §27)"},
        "units": {"pressure": punit, "speed": vunit, "length": lunit},
        "assumptions": [
            "Enclosed/partially-enclosed rigid building, MWFRS directional procedure.",
            "External wall pressures evaluated at mean roof height qh (simplified).",
            "Flat terrain (Kzt=1.0), sea level (Ke=1.0), Kd=0.85, G=0.85.",
            "Walls only — roof zones, C&C, and topography are in the full app.",
        ],
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/wind")
def wind(body: WindIn, request: Request):
    ip = _client_ip(request)
    run_no = _check_and_count(ip, "wind", WIND_FREE_RUNS)   # raises 429/403 as needed
    result = compute_wind(body)
    result["run"] = {"number": run_no, "free_limit": WIND_FREE_RUNS, "remaining": WIND_FREE_RUNS - run_no}
    return result


# ── Parametric portal-frame geometry ──────────────────────────────────────
class FrameIn(BaseModel):
    units: str = Field("si")
    span: float = Field(..., gt=0, le=200, description="transverse span S")
    eave_height: float = Field(..., gt=0, le=100)
    roof_pitch: float = Field(..., ge=0, le=45, description="degrees")
    bays: int = Field(..., ge=1, le=30, description="number of bays along length")
    bay_spacing: float = Field(..., gt=0, le=50)
    secondary: int = Field(2, ge=0, le=8, description="purlins per roof slope / girts per wall")

    @field_validator("units")
    @classmethod
    def _u(cls, v):
        v = v.lower()
        if v not in ("si", "imperial"):
            raise ValueError("units must be 'si' or 'imperial'")
        return v


def compute_frame(inp: FrameIn) -> dict:
    S, H, N, b = inp.span, inp.eave_height, inp.bays, inp.bay_spacing
    theta = math.radians(inp.roof_pitch)
    ridge_h = H + (S / 2.0) * math.tan(theta)
    frames = N + 1
    length = N * b

    nodes: list[list[float]] = []
    members: list[dict] = []

    def add(x, y, z) -> int:
        nodes.append([round(x, 3), round(y, 3), round(z, 3)])
        return len(nodes) - 1

    # Main portal frames
    main = []
    for k in range(frames):
        z = k * b
        bl, br = add(0, 0, z), add(S, 0, z)
        el, er = add(0, H, z), add(S, H, z)
        rg = add(S / 2.0, ridge_h, z)
        main.append((bl, br, el, er, rg))
        members += [
            {"i": bl, "j": el, "t": "column"}, {"i": br, "j": er, "t": "column"},
            {"i": el, "j": rg, "t": "rafter"}, {"i": rg, "j": er, "t": "rafter"},
        ]
    # Longitudinal eave + ridge beams
    for k in range(frames - 1):
        a, c = main[k], main[k + 1]
        members += [
            {"i": a[2], "j": c[2], "t": "eave"}, {"i": a[3], "j": c[3], "t": "eave"},
            {"i": a[4], "j": c[4], "t": "ridge"},
        ]

    # Secondary steel: purlins (roof) + girts (walls)
    if inp.secondary > 0:
        c = inp.secondary
        fr = [(i + 1) / (c + 1) for i in range(c)]          # evenly-spaced along each rafter
        hgts = [H * (i + 1) / (c + 1) for i in range(c)]     # evenly-spaced up each wall
        purl, girt = [], []
        for k in range(frames):
            z = k * b
            prow = []
            for f in fr:  # left slope
                prow.append(add(f * (S / 2.0), H + f * (ridge_h - H), z))
            for f in fr:  # right slope
                prow.append(add(S / 2.0 + f * (S / 2.0), ridge_h + f * (H - ridge_h), z))
            purl.append(prow)
            grow = []
            for h in hgts:
                grow.append(add(0, h, z))
                grow.append(add(S, h, z))
            girt.append(grow)
        for k in range(frames - 1):
            for idx in range(len(purl[k])):
                members.append({"i": purl[k][idx], "j": purl[k + 1][idx], "t": "purlin"})
            for idx in range(len(girt[k])):
                members.append({"i": girt[k][idx], "j": girt[k + 1][idx], "t": "girt"})

    lunit = "m" if inp.units == "si" else "ft"
    return {
        "nodes": nodes,
        "members": members,
        "meta": {
            "frames": frames, "span": S, "length": round(length, 2),
            "eave_height": H, "ridge_height": round(ridge_h, 3),
            "node_count": len(nodes), "member_count": len(members),
            "length_unit": lunit,
        },
    }


@app.post("/frame")
def frame(body: FrameIn, request: Request):
    ip = _client_ip(request)
    run_no = _check_and_count(ip, "frame", FRAME_FREE_RUNS)
    result = compute_frame(body)
    result["run"] = {"number": run_no, "free_limit": FRAME_FREE_RUNS, "remaining": FRAME_FREE_RUNS - run_no}
    return result

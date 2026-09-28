"""Physics ground truth from NHTSA sensor data (ISO-MME exports).

For each test we integrate structural accelerations to get the time of **maximum dynamic crush**
and the **dynamic crush depth**, a measured replacement for the visual ``max_deformation`` label,
which annotators could not place reliably (docs/ANNOTATION_QA.md):

- frontal barrier / side pole: the vehicle moves into a rigid obstacle. Its velocity towards the
  obstacle, v(t) = v0 + integral of a, reaches zero at maximum crush.
- side MDB: the moving barrier hits the standing vehicle. Crush grows while the barrier moves
  faster than the struck side, so maximum crush is where the two velocities meet.

Sign conventions differ between labs and sensor orientations, so each integrated signal is
oriented from the physics (an obstacle can only decelerate the impacting body, and the struck
vehicle can only be pushed away), not from metadata. Every derived value keeps the channels it
came from.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field

import numpy as np

HEADER = re.compile(r"^(.+?)\s*:(.*)$")


@dataclass
class Channel:
    number: str
    code: str
    header: dict
    t: np.ndarray  # s
    a: np.ndarray  # m/s^2

    @property
    def label(self) -> str:
        return f"{self.code} ({self.header.get('Comments', '').strip()})"


@dataclass
class IsoTest:
    test_no: int
    mme: dict
    channels: dict[str, str]  # number -> code (from the .chn index)
    zf: zipfile.ZipFile
    stem: str
    missing: list[str] = field(default_factory=list)

    def object_velocity(self, obj: int) -> float | None:
        v = self.mme.get(f"Velocity test object {obj}", "").strip()
        try:
            return float(v)
        except ValueError:
            return None

    def load(self, number: str) -> Channel | None:
        base = self.stem.split("\\")[-1]
        name = self.stem + "\\Channel\\" + base + "." + number
        try:
            text = self.zf.read(name).decode("latin-1")
        except KeyError:
            self.missing.append(number)
            return None
        return parse_channel(number, self.channels[number], text)


def parse_header(text: str) -> dict:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = HEADER.match(line.strip())
        if m:
            key = m.group(1).strip()
            out[key] = out[key] + " " + m.group(2).strip() if key in out else m.group(2).strip()
    return out


def parse_channel(number: str, code: str, text: str) -> Channel:
    lines = text.splitlines()
    hdr_lines, data = [], []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if ":" in s and not data:
            hdr_lines.append(s)
        else:
            try:
                data.append(float(s))
            except ValueError:
                continue
    hdr = parse_header("\n".join(hdr_lines))
    dt = float(hdr["Sampling interval"])
    t0 = float(hdr["Time of first sample"])
    a = np.asarray(data, dtype=float)
    if hdr.get("Unit", "").strip().lower() in ("g", "g's", "gs"):
        a = a * 9.80665
    return Channel(number, code, hdr, t0 + dt * np.arange(len(a)), a)


def open_iso(path, test_no: int) -> IsoTest:
    zf = zipfile.ZipFile(path)
    mme_name = next(n for n in zf.namelist() if n.lower().endswith(".mme"))
    chn_name = next(n for n in zf.namelist() if n.lower().endswith(".chn"))
    stem = mme_name.rsplit("\\", 1)[0]
    mme = parse_header(zf.read(mme_name).decode("latin-1"))
    chn = zf.read(chn_name).decode("latin-1")
    channels = dict(re.findall(r"Name of channel (\d+)\s*:(\S+)", chn))
    return IsoTest(test_no, mme, channels, zf, stem)


def lowpass(a: np.ndarray, dt: float, cutoff_hz: float = 100.0) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    b, c = butter(4, cutoff_hz / (0.5 / dt))
    return filtfilt(b, c, a)


def structure_channels(test: IsoTest, obj: int, axis: str, prefer: list[str]) -> list[Channel]:
    """Structural (position 0) acceleration channels of ``obj`` along ``axis``, best location
    first. ``prefer``: 4-letter ISO location codes in order of preference (e.g. VEHC, FORA)."""
    out = []
    for loc in prefer:
        for number, code in sorted(test.channels.items()):
            if (
                code.startswith(f"{obj}0")
                and code[2:6] == loc
                and code[12:14] == "AC"
                and code[14] == axis
            ):
                ch = test.load(number)
                if ch is not None and len(ch.a) > 100 and np.isfinite(ch.a).all():
                    out.append(ch)
        if out:
            return out  # all channels at the best available location (averaged later)
    return out


def mean_signal(chs: list[Channel], cutoff_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Common time base (intersection), offset removed (pre-impact mean), low-pass, averaged.

    Channels are sign-aligned to the first one before averaging (redundant sensors can be mounted
    mirrored); the overall sign is fixed later from the physics."""
    t0 = max(c.t[0] for c in chs)
    t1 = min(c.t[-1] for c in chs)
    dt = chs[0].t[1] - chs[0].t[0]
    t = np.arange(t0, t1, dt)
    sigs = []
    for c in chs:
        a = np.interp(t, c.t, c.a)
        pre = a[t < 0]
        if len(pre):
            a = a - pre.mean()
        sigs.append(lowpass(a, dt, cutoff_hz))
    ref = sigs[0]
    aligned = [s if np.dot(s, ref) >= 0 else -s for s in sigs]
    return t, np.mean(aligned, axis=0)


def integrate(t: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid from t = 0 (zero before impact)."""
    from scipy.integrate import cumulative_trapezoid

    v = cumulative_trapezoid(a, t, initial=0.0)
    return v - np.interp(0.0, t, v)


@dataclass
class Kinematics:
    test_no: int
    config: str
    t_max_crush_ms: float | None
    dynamic_crush_mm: float | None
    v0_mps: float | None
    channels: list[str]
    note: str = ""

    def as_row(self) -> dict:
        return {
            "test_no": self.test_no,
            "config": self.config,
            "t_max_crush_ms": self.t_max_crush_ms,
            "dynamic_crush_mm": self.dynamic_crush_mm,
            "v0_mps": self.v0_mps,
            "channels": "; ".join(self.channels),
            "note": self.note,
        }


def closing_end(t: np.ndarray, v: np.ndarray, v0: float, t_max: float = 0.2):
    """End of the closing phase: the first time the closing velocity reaches zero; if it never
    does (a yawing vehicle whose CG never quite matches the obstacle or barrier), the time of its
    minimum, provided that minimum is below 10 % of the impact speed. Returns (t, rule)."""
    w = (t > 0.005) & (t <= t_max)
    idx = np.where(w & (v <= 0))[0]
    if len(idx):
        i = idx[0]
        t_a, t_b, x_a, x_b = t[i - 1], t[i], v[i - 1], v[i]
        tc = float(t_a + (t_b - t_a) * x_a / (x_a - x_b)) if x_a != x_b else float(t_b)
        return tc, "zero-crossing"
    if not w.any():
        return None, "no data"
    j = np.where(w)[0][np.argmin(v[w])]
    if v[j] <= 0.1 * v0:
        return float(t[j]), "velocity-minimum"
    return None, f"closing velocity stays above 10 % of v0 (min {v[j]:.2f} m/s)"


def rigid_obstacle(test: IsoTest, config: str, cutoff_hz: float = 100.0) -> Kinematics:
    """Frontal barrier (X) or side pole (Y): v0 minus the integrated deceleration."""
    axis, prefer = (
        ("X", ["VEHC", "FORA", "SEAT", "SILL", "DOOR"])
        if config == "VTB"
        else ("Y", ["VEHC", "FORA", "SEAT"])
    )
    chs = structure_channels(test, 1, axis, prefer)
    v0 = test.object_velocity(1)
    if not chs or v0 is None:
        return Kinematics(test.test_no, config, None, None, v0, [], "no usable channel or v0")
    t, a = mean_signal(chs, cutoff_hz)
    dv = integrate(t, a)
    # orientation: the obstacle must decelerate the vehicle over the first 30 ms
    w = (t > 0) & (t < 0.03)
    sign = -1.0 if np.mean(dv[w]) > 0 else 1.0
    v = v0 + sign * dv
    tc, rule = closing_end(t, v, v0)
    crush = None
    if tc is not None:
        m = (t >= 0) & (t <= tc)
        crush = float(np.trapezoid(v[m], t[m]) * 1000)
    return Kinematics(
        test.test_no,
        config,
        None if tc is None else tc * 1000,
        crush,
        v0,
        [c.label for c in chs],
        rule,
    )


def side_mdb(test: IsoTest, cutoff_hz: float = 100.0) -> Kinematics:
    """Moving deformable barrier (object 1, X) into standing vehicle (object 2, Y)."""
    mdb = structure_channels(test, 1, "X", ["VEHC", "SILB"])
    veh = structure_channels(test, 2, "Y", ["VEHC", "FORA", "SEAT"])
    v0 = test.object_velocity(1)
    if not mdb or not veh or v0 is None:
        return Kinematics(
            test.test_no, "ITV", None, None, v0, [], "missing MDB/vehicle channel or v0"
        )
    tm, am = mean_signal(mdb, cutoff_hz)
    tv, av = mean_signal(veh, cutoff_hz)
    t = tm[(tm >= max(tm[0], tv[0])) & (tm <= min(tm[-1], tv[-1]))]
    dvm = integrate(t, np.interp(t, tm, am))
    dvv = integrate(t, np.interp(t, tv, av))
    w = (t > 0) & (t < 0.03)
    vm = v0 + (-1.0 if np.mean(dvm[w]) > 0 else 1.0) * dvm  # barrier slows down
    vv = (1.0 if np.mean(dvv[w]) > 0 else -1.0) * dvv  # vehicle is pushed away
    rel = vm - vv
    tc, rule = closing_end(t, rel, v0)
    crush = None
    if tc is not None:
        m = (t >= 0) & (t <= tc)
        crush = float(np.trapezoid(rel[m], t[m]) * 1000)
    return Kinematics(
        test.test_no,
        "ITV",
        None if tc is None else tc * 1000,
        crush,
        v0,
        [c.label for c in mdb + veh],
        rule,
    )


def kinematics(test: IsoTest, config: str, cutoff_hz: float = 100.0) -> Kinematics:
    if config == "ITV":
        return side_mdb(test, cutoff_hz)
    return rigid_obstacle(test, config, cutoff_hz)

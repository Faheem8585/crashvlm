import zipfile

import numpy as np
import pytest

from crashvlm.data.sensors import closing_end, kinematics, open_iso, parse_channel

DT = 1e-4


def channel_text(code, values, t0=-0.05, unit="m/(s*s)", comment="X"):
    head = [
        "Test object number          :1",
        f"Channel code                :{code}",
        f"Unit                        :{unit}",
        f"Sampling interval           :{DT}",
        f"Comments                    :{comment}",
        f"Time of first sample        :{t0:.6f}",
        f"Number of samples           :{len(values)}",
    ]
    return "\n".join(head + [f"{v:.6f}" for v in values]) + "\n"


def iso_zip(tmp_path, channels: dict[str, tuple[str, np.ndarray]], v0: float, test_no=1):
    stem = str(test_no)
    mme = "\n".join(
        [
            "Type of the test            :VEHICLE INTO BARRIER",
            f"Velocity test object 1      :{v0}",
            "Name of test object 1       :CAR",
        ]
    )
    chn = f"Number of channels          :{len(channels)}\n" + "\n".join(
        f"Name of channel {n}         :{code} / X" for n, (code, _) in channels.items()
    )
    path = tmp_path / "iso.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{stem}\\{stem}.mme", mme)
        z.writestr(f"{stem}\\Channel\\{stem}.chn", chn)
        for n, (code, vals) in channels.items():
            z.writestr(f"{stem}\\Channel\\{stem}.{n}", channel_text(code, vals))
    return path


def test_parse_channel_units_and_time():
    ch = parse_channel("001", "10VEHCCG0000ACXP", channel_text("x", [1.0, 2.0, 3.0], unit="G"))
    assert ch.t[0] == pytest.approx(-0.05) and ch.t[1] - ch.t[0] == pytest.approx(DT)
    assert ch.a[1] == pytest.approx(2 * 9.80665)


def test_rigid_barrier_constant_deceleration(tmp_path):
    # 15 m/s into a wall with -250 m/s^2 from t = 0 until 80 ms (rebound after v = 0):
    # v = 0 at 60 ms, crush = v0^2 / (2 a) = 450 mm
    t = -0.05 + DT * np.arange(3500)
    a = np.where((t >= 0) & (t < 0.08), -250.0, 0.0)
    path = iso_zip(
        tmp_path, {"001": ("10VEHCCG0000ACXP", a), "002": ("10VEHCCG0000ACXP", -a)}, 15.0
    )
    k = kinematics(open_iso(path, 1), "VTB")
    assert k.t_max_crush_ms == pytest.approx(60.0, abs=1.5)  # low-pass smooths the step
    assert k.dynamic_crush_mm == pytest.approx(450.0, rel=0.03)
    assert k.note == "zero-crossing"  # the mirrored redundant sensor was sign-aligned


def test_closing_end_falls_back_to_velocity_minimum():
    t = np.arange(0, 0.2, DT)
    v = 10 - 9.6 * np.minimum(t / 0.07, 1) + 2 * np.maximum(t - 0.07, 0)  # min 0.4 m/s at 70 ms
    tc, rule = closing_end(t, v, 10.0)
    assert rule == "velocity-minimum" and tc == pytest.approx(0.07, abs=1e-3)
    tc, rule = closing_end(t, v + 3, 10.0)
    assert tc is None and "stays above" in rule

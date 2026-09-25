import math

from crashvlm.data.catalog import build_rows, filter_tests, target_vehicle, to_float

ITV_VEHICLES = [
    {"vehicleNo": 1, "vehicleMake": "NHTSA", "vehicleModel": "DEFORMABLE IMPACTOR"},
    {
        "vehicleNo": 2,
        "vehicleMake": "CADILLAC",
        "vehicleModel": "XT6",
        "modelYear": 2020,
        "vehicleTestWeight": "2256",
        "vaxCrushDistance": "207",
    },
]
ITV_OCCUPANTS = [
    {
        "vehicleNo": 2,
        "occupantLocation": "LEFT FRONT SEAT",
        "occupantType": "ES-2RE",
        "headInjuryCriterion": "98",
        "millisecCliptDoraxRegionPeakAcceleration": "0.00",
    },
]


def test_zero_means_missing_only_for_listed_fields():
    assert math.isnan(to_float("0", "headInjuryCriterion"))
    assert to_float("0", "impactAngle") == 0
    assert math.isnan(to_float(None))
    assert to_float("56.65") == 56.65


def test_target_vehicle_skips_moving_barrier():
    assert target_vehicle(ITV_VEHICLES, ITV_OCCUPANTS)["vehicleMake"] == "CADILLAC"


def test_filter_tests():
    tests = [
        {
            "testNo": 1,
            "testType": "NEW CAR ASSESSMENT TEST",
            "testConfigurationKey": "VTB",
            "testDate": "2019-00-12",
        },
        {
            "testNo": 2,
            "testType": "EXPERIMENTAL NEW CAR ASSESSMENT TEST",
            "testConfigurationKey": "VTB",
            "testDate": "2019-00-12",
        },
        {
            "testNo": 3,
            "testType": "NEW CAR ASSESSMENT TEST",
            "testConfigurationKey": "VTB",
            "testDate": "2009-00-01",
        },
    ]
    got = filter_tests(tests, ["NEW CAR ASSESSMENT TEST"], ["VTB"], 2012)
    assert [t["testNo"] for t in got] == [1]


def test_build_rows_side_test():
    test = {
        "testNo": 10920,
        "testType": "NEW CAR ASSESSMENT TEST",
        "testConfigurationKey": "ITV",
        "testConfiguration": "IMPACTOR INTO VEHICLE",
        "testDate": "2020-00-01",
        "closingSpeed": 61.9,
    }
    media = {
        "videos": [
            {"originalFileName": "01 Overhead Wide View.wmv", "media": "u1", "fileSize": "1e7"},
            {
                "originalFileName": "07 Driver Dummy Front View (Onboard).wmv",
                "media": "u2",
                "fileSize": "1e7",
            },
        ]
    }
    row, occ = build_rows(test, media, ITV_VEHICLES, ITV_OCCUPANTS, max_mb=200)
    assert row["make"] == "CADILLAC" and row["crush_mm"] == 207
    assert row["exterior_url"] == "u1" and row["occupant_url"] == "u2"
    assert row["driver_hic"] == 98 and math.isnan(row["driver_chest_3ms_g"])
    assert occ[0]["seat"] == "driver"

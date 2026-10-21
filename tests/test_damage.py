from road_damage_colombia.domain.damage import Detection, cost, priority, severity


def test_a_big_pothole_in_the_lane_outranks_a_small_crack_on_the_side():
    pothole = Detection("D40", 0.9, (0.4, 0.6, 0.6, 0.8))
    crack = Detection("D00", 0.9, (0.0, 0.1, 0.05, 0.2))
    assert pothole.in_lane and not crack.in_lane
    assert severity(pothole) > 5 * severity(crack)


def test_priority_levels_and_treatments():
    assert priority([])["level"] == "none"
    p = priority([Detection("D40", 0.9, (0.4, 0.6, 0.6, 0.8)), Detection("D10", 0.5, (0.1, 0.5, 0.3, 0.55))])
    assert p["level"] == "urgent"
    assert p["treatments"]["pothole repair"]["count"] == 1 and "crack sealing" in p["treatments"]
    assert priority([Detection("D00", 0.4, (0.0, 0.0, 0.1, 0.1))])["level"] == "low"


def test_no_cost_without_prices():
    t = priority([Detection("D20", 0.8, (0.3, 0.5, 0.5, 0.7))])["treatments"]
    assert cost(t, None) is None
    assert cost(t, {"patching": 1000}) == {"patching": 40.0}

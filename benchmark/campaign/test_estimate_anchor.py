from .analysis import _select_anchor, method_class


def test_exact_observation_beats_same_shape_with_different_cost_distribution():
    case = dict(id="negative", family="base", kind="single", objective="optional",
                rows=128, cols=160, limit=-.25, metadata={})
    other = {**case, "id": "positive", "limit": .5}
    def anchor(value, seconds):
        return dict(case_id=value["id"], case=value, method="scipy:capped",
                    method_class=method_class("scipy:capped"), seconds=seconds)
    earlier = anchor(other, .001)
    exact = anchor(case, .01)
    chosen, quality = _select_anchor(case, "scipy:capped", [earlier, exact])
    assert chosen is exact
    assert quality == "observed_case_method"

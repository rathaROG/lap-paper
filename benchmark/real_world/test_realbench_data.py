"""Regression checks for the configured-size versus actual-network-size failure."""
from types import SimpleNamespace
import unittest

import torch

import realbench_data as data


def fake_detector(config=768, runtime=768, optimized=768):
    module = torch.nn.Identity()
    model = SimpleNamespace(
        model_config=SimpleNamespace(resolution=config),
        model=SimpleNamespace(resolution=runtime, inference_model=module),
        _optimized_resolution=optimized, _is_optimized_for_inference=True)
    return SimpleNamespace(model=model)


class ResolutionRuntimeTests(unittest.TestCase):
    def test_configuration_cannot_hide_wrong_runtime_or_stale_optimization(self):
        for detector in (fake_detector(runtime=384), fake_detector(optimized=384)):
            with self.subTest(runtime=detector.model.model.resolution,
                              optimized=detector.model._optimized_resolution):
                with self.assertRaisesRegex(RuntimeError, "requested/runtime resolution mismatch"):
                    data._rf_runtime_resolution(detector, 768)

    def test_actual_network_tensor_must_match_requested_size(self):
        detector = fake_detector()
        with self.assertRaisesRegex(RuntimeError, "observed"):
            with data._observe_rf_input_shapes(detector, 768):
                detector.model.model.inference_model(torch.zeros(1, 3, 32, 32))
        self.assertEqual(len(detector.model.model.inference_model._forward_pre_hooks), 0)

    def test_observation_requires_network_execution_and_removes_hook(self):
        detector = fake_detector()
        with self.assertRaisesRegex(RuntimeError, "not called"):
            with data._observe_rf_input_shapes(detector, 768):
                pass
        self.assertEqual(len(detector.model.model.inference_model._forward_pre_hooks), 0)
        with data._observe_rf_input_shapes(detector, 768) as observed:
            detector.model.model.inference_model(torch.zeros(1, 3, 768, 768))
        self.assertEqual(observed, [[1, 3, 768, 768]])
        self.assertEqual(len(detector.model.model.inference_model._forward_pre_hooks), 0)

    def test_per_call_settings_do_not_mutate_defaults(self):
        original = dict(data.RF_SETTINGS)
        changed = data._rf_settings(384, .4)
        self.assertEqual((changed["imgsz"], changed["conf"]), (384, .4))
        self.assertEqual(data.RF_SETTINGS, original)
        self.assertEqual((data._rf_settings(768, .2)["imgsz"], data._rf_settings(768, .2)["conf"]), (768, .2))


if __name__ == "__main__":
    unittest.main()

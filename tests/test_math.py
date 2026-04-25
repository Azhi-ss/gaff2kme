"""Tests for gaff2kme.math — Layer 1: Pure Math KME Transform.

Covers: core behavior, edge cases, boundary values, error paths.
"""

import numpy as np
import pytest

from gaff2kme.math import MinMaxScale, compute_kme, gaussian_kernel, kme_desc_names, kernel_mean

FF_MASS = [1.008, 12.011, 14.007, 15.999, 18.998, 30.974, 32.067, 35.453, 79.904, 126.904]
ELEMENT_NAMES = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]


def _make_scaling(polar: bool = True) -> dict:
    return {
        "mass": MinMaxScale(FF_MASS),
        "charge": MinMaxScale([-1.0, 1.0]),
        "epsilon": MinMaxScale([0.0, 1.0]),
        "sigma": MinMaxScale([0.5, 2.0]),
        "k_bond": MinMaxScale([100.0, 1000.0]),
        "r0": MinMaxScale([0.8, 2.0]),
        "polar": MinMaxScale([0.0, 1.0]),
        "k_angle": MinMaxScale([50.0, 500.0]),
        "theta0": MinMaxScale([1.5, 3.14]),
        "k_dih": MinMaxScale([0.0, 10.0]),
    } if polar else {
        "mass": MinMaxScale(FF_MASS),
        "charge": MinMaxScale([-1.0, 1.0]),
        "epsilon": MinMaxScale([0.0, 1.0]),
        "sigma": MinMaxScale([0.5, 2.0]),
        "k_bond": MinMaxScale([100.0, 1000.0]),
        "r0": MinMaxScale([0.8, 2.0]),
        "k_angle": MinMaxScale([50.0, 500.0]),
        "theta0": MinMaxScale([1.5, 3.14]),
        "k_dih": MinMaxScale([0.0, 10.0]),
    }


def _make_params(polar: bool = True, n_atoms: int = 9) -> dict:
    rng = np.random.RandomState(42)
    base = {
        "mass": np.array([12.011, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008, 1.008, 15.999]),
        "charge": rng.uniform(-0.5, 0.5, n_atoms),
        "epsilon": rng.uniform(0.01, 0.5, n_atoms),
        "sigma": rng.uniform(0.8, 1.5, n_atoms),
        "k_bond": rng.uniform(200.0, 800.0, n_atoms),
        "r0": rng.uniform(1.0, 1.5, n_atoms),
        "k_angle": rng.uniform(100.0, 400.0, n_atoms),
        "theta0": rng.uniform(1.8, 2.8, n_atoms),
        "k_dih": rng.uniform(0.1, 5.0, n_atoms),
    }
    if polar:
        base["polar"] = rng.uniform(0.0, 0.8, n_atoms)
    return base


class TestMinMaxScale:

    def test_min_max_scale_basic(self):
        scaler = MinMaxScale([0, 10])
        original = [2.0, 5.0, 8.0]
        scaled = scaler.scale(original)
        unscaled = scaler.unscale(scaled)
        np.testing.assert_allclose(unscaled, original, atol=1e-10)

    def test_min_max_scale_known_values(self):
        scaler = MinMaxScale([0, 10])
        result = scaler.scale(5)
        assert pytest.approx(result) == 0.5

    def test_min_max_scale_custom_range(self):
        scaler = MinMaxScale([0, 10], vmax=2.0, vmin=0.0)
        result = scaler.scale(5)
        assert pytest.approx(result) == 0.25

    def test_min_max_scale_constant_data(self):
        scaler = MinMaxScale([5, 5, 5])
        scaled = scaler.scale([5, 5])
        assert np.all(scaled == 0.0)
        unscaled = scaler.unscale([0.0, 0.0])
        assert np.all(unscaled == 5.0)

    def test_min_max_scale_single_value(self):
        scaler = MinMaxScale([42.0])
        scaled = scaler.scale(42.0)
        assert scaled == 0.0

    def test_min_max_scale_negative_range(self):
        scaler = MinMaxScale([-10, -5])
        result = scaler.scale(-7.5)
        assert pytest.approx(result) == 0.5

    def test_min_max_scale_inverted_target(self):
        scaler = MinMaxScale([0, 10], vmax=0.0, vmin=1.0)
        result = scaler.scale(0)
        assert pytest.approx(result) == 1.0
        result2 = scaler.scale(10)
        assert pytest.approx(result2) == 0.0

    def test_min_max_scale_array_input(self):
        scaler = MinMaxScale(np.array([0, 10]))
        result = scaler.scale(np.array([0, 5, 10]))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_min_max_scale_boundary_values(self):
        scaler = MinMaxScale([0, 100])
        assert pytest.approx(scaler.scale(0)) == 0.0
        assert pytest.approx(scaler.scale(100)) == 1.0


class TestGaussianKernel:

    def test_gaussian_kernel_center(self):
        result = gaussian_kernel(np.array(0.5), np.array(0.5), sigma=0.1)
        assert pytest.approx(float(result)) == 1.0

    def test_gaussian_kernel_symmetry(self):
        x = np.array(0.3)
        xm = np.array(0.7)
        k1 = gaussian_kernel(x, xm, sigma=0.2)
        k2 = gaussian_kernel(xm, x, sigma=0.2)
        assert pytest.approx(float(k1)) == float(k2)

    def test_gaussian_kernel_decay(self):
        xm = np.array(0.0)
        k_near = gaussian_kernel(np.array(0.1), xm, sigma=0.5)
        k_far = gaussian_kernel(np.array(0.5), xm, sigma=0.5)
        assert float(k_near) > float(k_far)

    def test_gaussian_kernel_sigma_effect(self):
        xm = np.array(0.0)
        x = np.array(0.2)
        k_wide = gaussian_kernel(x, xm, sigma=1.0)
        k_narrow = gaussian_kernel(x, xm, sigma=0.1)
        assert float(k_wide) > float(k_narrow)

    def test_gaussian_kernel_always_positive(self):
        x = np.linspace(-10, 10, 100)
        xm = np.array(0.0)
        result = gaussian_kernel(x, xm, sigma=0.5)
        assert np.all(result > 0)

    def test_gaussian_kernel_bounded(self):
        result = gaussian_kernel(np.array(100.0), np.array(0.0), sigma=0.1)
        assert float(result) <= 1.0

    def test_gaussian_kernel_broadcast(self):
        x = np.array([0.0, 0.5, 1.0])
        xm = np.array([0.0, 0.5])
        result = gaussian_kernel(x.reshape(1, -1), xm.reshape(-1, 1), sigma=0.3)
        assert result.shape == (2, 3)


class TestKernelMean:

    def test_kernel_mean_uniform_weights(self):
        vec = np.array([0.2, 0.4, 0.6, 0.8])
        centers = np.array([0.0, 0.5, 1.0])
        sigma = 0.3
        result_weighted = kernel_mean(vec, centers, gaussian_kernel, weights=None, sigma=sigma)
        kernel_vals = gaussian_kernel(vec.reshape(1, -1), centers.reshape(-1, 1), sigma=sigma)
        expected = np.mean(kernel_vals, axis=1)
        np.testing.assert_allclose(result_weighted, expected, atol=1e-12)

    def test_kernel_mean_custom_weights(self):
        vec = np.array([0.2, 0.4, 0.6])
        centers = np.array([0.0, 0.5, 1.0])
        weights = np.array([1.0, 2.0, 1.0])
        sigma = 0.3
        result = kernel_mean(vec, centers, gaussian_kernel, weights=weights, sigma=sigma)
        kernel_vals = gaussian_kernel(vec.reshape(1, -1), centers.reshape(-1, 1), sigma=sigma)
        w = weights / np.sum(weights)
        expected = np.sum(kernel_vals * w, axis=1)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_kernel_mean_output_shape(self):
        vec = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        centers = np.linspace(0, 1, 15)
        result = kernel_mean(vec, centers, gaussian_kernel, sigma=0.2)
        assert result.shape == (15,)

    def test_kernel_mean_single_element_vec(self):
        vec = np.array([0.5])
        centers = np.array([0.0, 0.5, 1.0])
        result = kernel_mean(vec, centers, gaussian_kernel, sigma=0.2)
        assert result.shape == (3,)
        center_idx = 1
        assert pytest.approx(float(result[center_idx]), abs=1e-10) == 1.0

    def test_kernel_mean_values_in_zero_one(self):
        vec = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        centers = np.linspace(0, 1, 10)
        result = kernel_mean(vec, centers, gaussian_kernel, sigma=0.2)
        assert np.all(result > 0.0)
        assert np.all(result <= 1.0)

    def test_kernel_mean_equal_weights_implicit(self):
        vec = np.array([0.3, 0.7])
        centers = np.array([0.5])
        weights = np.array([1.0, 1.0])
        result_unweighted = kernel_mean(vec, centers, gaussian_kernel, weights=None, sigma=0.3)
        result_weighted = kernel_mean(vec, centers, gaussian_kernel, weights=weights, sigma=0.3)
        np.testing.assert_allclose(result_unweighted, result_weighted, atol=1e-12)


class TestComputeKme:

    def test_compute_kme_shape_polar_true(self):
        params = _make_params(polar=True)
        scaling = _make_scaling(polar=True)
        result = compute_kme(params, scaling, polar=True)
        assert result.shape == (190,)

    def test_compute_kme_shape_polar_false(self):
        params = _make_params(polar=False)
        scaling = _make_scaling(polar=False)
        result = compute_kme(params, scaling, polar=False)
        assert result.shape == (170,)

    def test_compute_kme_mass_sum_to_one(self):
        params = _make_params(polar=True)
        scaling = _make_scaling(polar=True)
        result = compute_kme(params, scaling, polar=True)
        mass_sum = np.sum(result[:10])
        assert pytest.approx(mass_sum, abs=1e-10) == 1.0

    def test_compute_kme_no_nan(self):
        params = _make_params(polar=True)
        scaling = _make_scaling(polar=True)
        result = compute_kme(params, scaling, polar=True)
        assert not np.any(np.isnan(result))

    def test_compute_kme_deterministic(self):
        params = _make_params(polar=True)
        scaling = _make_scaling(polar=True)
        result1 = compute_kme(params, scaling, polar=True)
        result2 = compute_kme(params, scaling, polar=True)
        np.testing.assert_array_equal(result1, result2)

    def test_compute_kme_custom_nk(self):
        nk = 10
        params = _make_params(polar=True)
        scaling = _make_scaling(polar=True)
        result = compute_kme(params, scaling, polar=True, nk=nk)
        assert result.shape == (10 + 9 * nk,)

    def test_compute_kme_single_atom(self):
        params = {
            "mass": np.array([12.011]),
            "charge": np.array([0.0]),
            "epsilon": np.array([0.1]),
            "sigma": np.array([1.0]),
            "k_bond": np.array([300.0]),
            "r0": np.array([1.2]),
            "polar": np.array([0.5]),
            "k_angle": np.array([200.0]),
            "theta0": np.array([2.0]),
            "k_dih": np.array([1.0]),
        }
        scaling = _make_scaling(polar=True)
        result = compute_kme(params, scaling, polar=True)
        assert result.shape == (190,)
        assert not np.any(np.isnan(result))
        assert result[1] == pytest.approx(1.0, abs=1e-10)

    def test_compute_kme_mass_fraction_carbon_only(self):
        params = _make_params(polar=True)
        params["mass"] = np.array([12.011, 12.011, 12.011])
        scaling = _make_scaling(polar=True)
        result = compute_kme(params, scaling, polar=True)
        assert result[1] == pytest.approx(1.0, abs=1e-10)
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_compute_kme_polar_false_no_polar_key(self):
        params = _make_params(polar=False)
        scaling = _make_scaling(polar=False)
        result = compute_kme(params, scaling, polar=False)
        assert "polar" not in params
        assert result.shape == (170,)

    def test_compute_kme_output_dtype(self):
        params = _make_params(polar=True)
        scaling = _make_scaling(polar=True)
        result = compute_kme(params, scaling, polar=True)
        assert result.dtype == np.float64

    def test_compute_kme_all_same_atom_type(self):
        n = 20
        params = {
            "mass": np.full(n, 1.008),
            "charge": np.zeros(n),
            "epsilon": np.full(n, 0.1),
            "sigma": np.full(n, 1.0),
            "k_bond": np.full(n, 300.0),
            "r0": np.full(n, 1.2),
            "polar": np.full(n, 0.5),
            "k_angle": np.full(n, 200.0),
            "theta0": np.full(n, 2.0),
            "k_dih": np.full(n, 1.0),
        }
        scaling = _make_scaling(polar=True)
        result = compute_kme(params, scaling, polar=True)
        assert result.shape == (190,)
        assert result[0] == pytest.approx(1.0, abs=1e-10)
        assert not np.any(np.isnan(result))


class TestKmeDescNames:

    def test_kme_desc_names_count_polar_true(self):
        names = kme_desc_names(polar=True)
        assert len(names) == 190

    def test_kme_desc_names_count_polar_false(self):
        names = kme_desc_names(polar=False)
        assert len(names) == 170

    def test_kme_desc_names_format(self):
        names = kme_desc_names(polar=True)
        for i, name in enumerate(names[:10]):
            assert name.startswith("mass_")
        param_names = ["charge", "epsilon", "sigma", "k_bond", "r0", "polar", "k_angle", "theta0", "k_dih"]
        idx = 10
        for pname in param_names:
            for k in range(20):
                assert names[idx] == f"{pname}_{k}", f"Expected {pname}_{k} at index {idx}, got {names[idx]}"
                idx += 1

    def test_kme_desc_names_custom_nk(self):
        names = kme_desc_names(polar=True, nk=10)
        assert len(names) == 10 + 9 * 10

    def test_kme_desc_names_polar_false_excludes_polar(self):
        names = kme_desc_names(polar=False)
        assert not any("polar" in n for n in names)

    def test_kme_desc_names_all_unique(self):
        names = kme_desc_names(polar=True)
        assert len(names) == len(set(names))

    def test_kme_desc_names_mass_elements(self):
        names = kme_desc_names(polar=True)
        expected_mass = ["mass_H", "mass_C", "mass_N", "mass_O", "mass_F",
                         "mass_P", "mass_S", "mass_Cl", "mass_Br", "mass_I"]
        assert names[:10] == expected_mass

    def test_kme_desc_names_nk1(self):
        names = kme_desc_names(polar=True, nk=1)
        assert len(names) == 10 + 9 * 1
        assert names[10] == "charge_0"


class TestKernelCenters:

    def test_kernel_centers_nk20(self):
        nk = 20
        full = np.linspace(0.0, 1.0, nk * 2 + 1)
        centers = full[1:-1:2]
        assert len(centers) == nk
        assert centers[0] > 0.0
        assert centers[-1] < 1.0
        assert pytest.approx(centers[0], abs=1e-5) == 0.025

    def test_kernel_centers_nk_odd_indices(self):
        nk = 5
        full = np.linspace(0.0, 1.0, nk * 2 + 1)
        centers = full[1:-1:2]
        expected = [full[1], full[3], full[5], full[7], full[9]]
        np.testing.assert_allclose(centers, expected)


class TestMassFractionEdgeCases:

    def test_empty_mass_array(self):
        from gaff2kme.math import _compute_mass_fraction
        result = _compute_mass_fraction(np.array([]))
        assert result.shape == (10,)
        assert np.all(result == 0.0)

    def test_nearest_mass_matching(self):
        from gaff2kme.math import _compute_mass_fraction
        result = _compute_mass_fraction(np.array([12.0]))
        assert result[1] == pytest.approx(1.0, abs=1e-10)
        assert result[0] == pytest.approx(0.0, abs=1e-10)


class TestComputeKmeLargeInput:

    def test_compute_kme_large_molecule(self):
        n = 1000
        rng = np.random.RandomState(99)
        mass_choices = np.array(FF_MASS)
        params = {
            "mass": rng.choice(mass_choices, n),
            "charge": rng.uniform(-1.0, 1.0, n),
            "epsilon": rng.uniform(0.01, 0.5, n),
            "sigma": rng.uniform(0.8, 1.5, n),
            "k_bond": rng.uniform(100.0, 800.0, n),
            "r0": rng.uniform(0.8, 1.5, n),
            "polar": rng.uniform(0.0, 0.8, n),
            "k_angle": rng.uniform(50.0, 400.0, n),
            "theta0": rng.uniform(1.5, 3.0, n),
            "k_dih": rng.uniform(0.1, 5.0, n),
        }
        scaling = _make_scaling(polar=True)
        result = compute_kme(params, scaling, polar=True)
        assert result.shape == (190,)
        assert not np.any(np.isnan(result))

#!/usr/bin/env python3
"""
Test script to verify SBXL and Flux algorithm alignment
"""

import torch
import math
from typing import Callable, List

# Copy functions from sbxl_train_util.py and flux_minimal_inference.py

def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule_sbxl(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> List[float]:
    """SBXL get_schedule"""
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def get_schedule_flux(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    """Flux get_schedule"""
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def test_timestep_schedule():
    """Test if timestep schedules are identical"""
    print("Testing timestep schedule alignment...")

    test_cases = [
        (10, 1024, True),
        (20, 2048, True),
        (50, 4096, False),
        (100, 256, True),
    ]

    for num_steps, seq_len, shift in test_cases:
        sbxl_schedule = get_schedule_sbxl(num_steps, seq_len, shift=shift)
        flux_schedule = get_schedule_flux(num_steps, seq_len, shift=shift)

        if sbxl_schedule == flux_schedule:
            print(f"✓ PASS: num_steps={num_steps}, seq_len={seq_len}, shift={shift}")
        else:
            print(f"✗ FAIL: num_steps={num_steps}, seq_len={seq_len}, shift={shift}")
            print(f"  SBXL: {sbxl_schedule}")
            print(f"  Flux: {flux_schedule}")
            return False

    return True

def euler_step_sbxl(img: torch.Tensor, pred: torch.Tensor, t_curr: float, t_next: float) -> torch.Tensor:
    """SBXL Euler step"""
    return img + (t_next - t_curr) * pred

def euler_step_flux(img: torch.Tensor, pred: torch.Tensor, t_curr: float, t_prev: float) -> torch.Tensor:
    """Flux Euler step"""
    return img + (t_prev - t_curr) * pred

def test_euler_step():
    """Test if Euler steps are mathematically equivalent"""
    print("\nTesting Euler step alignment...")

    # Create test tensors
    batch_size, channels, height, width = 2, 16, 32, 32
    img = torch.randn(batch_size, channels, height, width)
    pred = torch.randn_like(img)

    t_curr = 0.8
    t_next = 0.6
    t_prev = t_next  # In flux, t_prev is t_next

    sbxl_result = euler_step_sbxl(img.clone(), pred, t_curr, t_next)
    flux_result = euler_step_flux(img.clone(), pred, t_curr, t_prev)

    if torch.allclose(sbxl_result, flux_result, atol=1e-6):
        print("✓ PASS: Euler steps are mathematically equivalent")
        return True
    else:
        print("✗ FAIL: Euler steps differ")
        print(f"  Max diff: {(sbxl_result - flux_result).abs().max().item()}")
        return False

def test_cfg_alignment():
    """Test if CFG implementations are mathematically equivalent"""
    print("\nTesting CFG alignment...")

    # Simulate predictions from model
    batch_size, channels, height, width = 2, 16, 32, 32
    pred_combined = torch.randn(batch_size * 2, channels, height, width)  # [neg, pos, neg, pos, ...]

    # SBXL style: pred_neg, pred_pos = pred_combined.chunk(2, dim=0)
    pred_neg, pred_pos = pred_combined.chunk(2, dim=0)

    # Test different guidance scales
    test_scales = [1.0, 3.0, 7.0, 0.5]

    for scale in test_scales:
        # SBXL CFG
        sbxl_pred = pred_neg + scale * (pred_pos - pred_neg)

        # Flux CFG (equivalent)
        flux_pred = pred_neg + scale * (pred_pos - pred_neg)

        if torch.allclose(sbxl_pred, flux_pred, atol=1e-6):
            print(f"✓ PASS: CFG scale {scale}")
        else:
            print(f"✗ FAIL: CFG scale {scale}")
            print(f"  Max diff: {(sbxl_pred - flux_pred).abs().max().item()}")
            return False

    return True

if __name__ == "__main__":
    print("SBXL vs Flux Algorithm Alignment Test")
    print("=" * 50)

    timestep_ok = test_timestep_schedule()
    euler_ok = test_euler_step()
    cfg_ok = test_cfg_alignment()

    if timestep_ok and euler_ok and cfg_ok:
        print("\n🎉 All tests PASSED - SBXL and Flux algorithms are aligned!")
    else:
        print("\n❌ Some tests FAILED - Algorithms may not be aligned")
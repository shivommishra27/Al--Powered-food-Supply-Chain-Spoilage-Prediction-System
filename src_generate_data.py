"""Generate a synthetic cold-chain / IoT dataset for food spoilage prediction.

Real sensor data is rarely public, so this simulates a realistic process:
spoilage speeds up exponentially as temperature drifts from the ideal, and is
worsened by humidity, temperature swings, ethylene, door openings and
poor packaging.
"""
import argparse

import numpy as np
import pandas as pd

from .config import DATA_PATH, PACKAGING, PRODUCTS, RANDOM_STATE


def generate(n_samples: int = 8000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(PRODUCTS)
    rows = []
    for _ in range(n_samples):
        product = rng.choice(names)
        base_life, ideal_temp, sens = PRODUCTS[product]
        packaging = rng.choice(list(PACKAGING))

        avg_temp = ideal_temp + rng.normal(1.0, 3.0)          # drift from ideal
        temp_std = abs(rng.normal(1.2, 1.0))
        humidity = float(np.clip(rng.normal(80, 10), 30, 100))
        transit_hours = float(rng.gamma(3.0, 10.0))            # ~30 h average
        days_since_harvest = float(rng.uniform(0, base_life * 0.6))
        co2 = float(np.clip(rng.normal(600, 250), 300, 2500))
        ethylene = float(abs(rng.normal(8, 6)))
        door_open = int(rng.poisson(3))
        vibration = float(abs(rng.normal(0.4, 0.25)))

        # --- ground-truth decay simulation ---
        temp_factor = np.exp(sens * abs(avg_temp - ideal_temp) / 8.0)
        swing_factor = 1 + 0.15 * temp_std
        hum_factor = 1 + 0.015 * max(humidity - 85, 0) + 0.01 * max(60 - humidity, 0)
        ethylene_factor = 1 + ethylene / 60.0
        door_factor = 1 + 0.02 * door_open
        vib_factor = 1 + 0.2 * vibration
        co2_factor = 1 + 0.0001 * max(co2 - 1000, 0)
        pack_factor = PACKAGING[packaging]

        rate = (temp_factor * swing_factor * hum_factor * ethylene_factor *
                door_factor * vib_factor * co2_factor * pack_factor)
        elapsed_days = days_since_harvest + transit_hours / 24.0
        remaining = base_life - elapsed_days * rate + rng.normal(0, base_life * 0.03)
        remaining = max(remaining, 0.0)

        rows.append({
            "product": product,
            "packaging": packaging,
            "avg_temp_c": round(avg_temp, 2),
            "temp_std_c": round(temp_std, 2),
            "humidity_pct": round(humidity, 1),
            "transit_hours": round(transit_hours, 1),
            "days_since_harvest": round(days_since_harvest, 2),
            "co2_ppm": round(co2, 0),
            "ethylene_ppm": round(ethylene, 2),
            "door_open_events": door_open,
            "vibration_g": round(vibration, 3),
            "remaining_shelf_life_days": round(remaining, 2),
            "spoiled": int(remaining <= 0.0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic supply-chain data")
    parser.add_argument("--n", type=int, default=8000, help="number of rows")
    parser.add_argument("--out", type=str, default=str(DATA_PATH))
    args = parser.parse_args()

    df = generate(args.n)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")
    print(f"Spoiled ratio: {df['spoiled'].mean():.2%}")


if __name__ == "__main__":
    main()

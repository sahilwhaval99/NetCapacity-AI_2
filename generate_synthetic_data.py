"""
Synthetic Network Traffic Telemetry Generator for NetCapacity AI
================================================================
Generates realistic, deterministic, and clearly labeled synthetic network datasets
for third-year B.Tech AI & DS academic evaluation and testing.

Each dataset models:
1. Diurnal (24-hour) enterprise/ISP traffic seasonality (business hours peak, evening streaming peak, night trough).
2. Weekly (7-day) seasonality (weekday vs. weekend traffic profile).
3. Secular organic bandwidth growth trend over the observation window.
4. Stochastic micro-bursts (scheduled cloud backups, software patch rollouts, live events).
5. Correlated telemetry metrics (active_users, latency_ms, packet_loss_pct).
6. Controlled missing-value gaps (~1% NaNs) to test the preprocessing pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_hourly_dataset(
    filename: str,
    days: int = 60,
    base_mbps: float = 420.0,
    growth_mbps: float = 165.0,
    link_capacity_mbps: float = 1000.0,
    profile: str = "campus",
    seed: int = 42,
    inject_missing_rate: float = 0.01,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = days * 24
    timestamps = pd.date_range(start="2026-08-01 00:00:00", periods=periods, freq="h")

    hours = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    t_norm = np.linspace(0.0, 1.0, periods)

    # 1. Secular linear + slight quadratic user/traffic growth
    trend = base_mbps + growth_mbps * (0.75 * t_norm + 0.25 * (t_norm ** 1.4))

    # 2. Diurnal seasonality (24-hour pattern)
    if profile == "campus":
        work_peak = 240.0 * np.exp(-0.5 * ((hours - 13.5) / 3.2) ** 2)
        evening_peak = 135.0 * np.exp(-0.5 * ((hours - 21.0) / 2.0) ** 2)
        night_trough = -140.0 * np.exp(-0.5 * ((hours - 3.5) / 2.2) ** 2)
        diurnal = work_peak + evening_peak + night_trough
        weekend_factor = np.where(dow >= 5, 0.68, 1.0)
    else:
        day_usage = 290.0 * np.exp(-0.5 * ((hours - 14.0) / 4.0) ** 2)
        prime_time = 620.0 * np.exp(-0.5 * ((hours - 21.0) / 2.3) ** 2)
        night_trough = -310.0 * np.exp(-0.5 * ((hours - 4.0) / 2.4) ** 2)
        diurnal = day_usage + prime_time + night_trough
        weekend_factor = np.where(dow >= 5, 1.14, 1.0)

    # 3. Autoregressive AR(1) correlated noise + Gaussian innovation
    ar_noise = np.zeros(periods)
    innovation_std = 22.0 if profile == "campus" else 45.0
    innovations = rng.normal(0.0, innovation_std, size=periods)
    for i in range(1, periods):
        ar_noise[i] = 0.55 * ar_noise[i - 1] + innovations[i]

    # 4. Scheduled backup / patch burst anomalies (~1.8% of hours)
    bursts = np.zeros(periods)
    burst_indices = rng.choice(periods, size=max(5, int(periods * 0.018)), replace=False)
    burst_magnitudes = rng.uniform(0.08 * link_capacity_mbps, 0.19 * link_capacity_mbps, size=len(burst_indices))
    bursts[burst_indices] = burst_magnitudes

    bandwidth = (trend + diurnal) * weekend_factor + ar_noise + bursts
    bandwidth = np.clip(bandwidth, 45.0, link_capacity_mbps * 0.99)

    # 5. Correlated telemetry columns
    utilization_ratio = bandwidth / link_capacity_mbps
    active_users = np.round(
        (bandwidth / (0.38 if profile == "campus" else 0.52))
        + rng.normal(0, 35, size=periods)
    ).astype(int)
    active_users = np.clip(active_users, 50, 25000)

    # Non-linear queuing delay latency (M/M/1 style hockey-stick when utilization > 72%)
    base_latency = 8.5 if profile == "campus" else 14.0
    excess_util = np.maximum(0.0, (utilization_ratio - 0.72) * 100.0)
    queuing_delay = np.where(
        utilization_ratio > 0.72,
        (excess_util ** 1.65) * 0.45,
        utilization_ratio * 6.0,
    )
    latency_ms = np.round(
        np.clip(base_latency + queuing_delay + rng.normal(0, 1.2, size=periods), 2.0, 250.0),
        2,
    )

    # Packet loss rises sharply when utilization exceeds 80%
    packet_loss_pct = np.where(
        utilization_ratio > 0.80,
        (utilization_ratio - 0.80) * 14.5 + rng.uniform(0.0, 0.35, size=periods),
        rng.uniform(0.0, 0.06, size=periods),
    )
    packet_loss_pct = np.round(np.clip(packet_loss_pct, 0.0, 12.0), 3)

    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "bandwidth_mbps": np.round(bandwidth, 2),
            "active_users": active_users,
            "latency_ms": latency_ms,
            "packet_loss_pct": packet_loss_pct,
            "allocated_capacity_mbps": link_capacity_mbps,
            "data_source": f"SYNTHETIC_{profile.upper()}_SEED_{seed}",
        }
    )

    if inject_missing_rate > 0:
        n_missing = int(periods * inject_missing_rate)
        missing_idx = rng.choice(np.arange(10, periods - 10), size=n_missing, replace=False)
        df.loc[missing_idx, "bandwidth_mbps"] = np.nan

    df.to_csv(filename, index=False)
    return df


def generate_daily_dataset(
    filename: str,
    days: int = 180,
    base_mbps: float = 2100.0,
    growth_mbps: float = 1450.0,
    link_capacity_mbps: float = 5000.0,
    seed: int = 108,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start="2026-04-01", periods=days, freq="D")
    dow = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    t_norm = np.linspace(0.0, 1.0, days)

    trend = base_mbps + growth_mbps * (0.8 * t_norm + 0.2 * (t_norm ** 2))
    weekly = np.where(dow < 5, 320.0, -410.0) + 95.0 * np.sin(2 * np.pi * dow / 7.0)
    monthly_cycle = 140.0 * np.sin(2 * np.pi * day_of_year / 30.5)
    noise = rng.normal(0.0, 110.0, size=days)

    bandwidth = np.clip(trend + weekly + monthly_cycle + noise, 600.0, link_capacity_mbps * 0.98)
    utilization_ratio = bandwidth / link_capacity_mbps
    active_users = np.round(bandwidth * 3.2 + rng.normal(0, 180, size=days)).astype(int)
    latency_ms = np.round(11.0 + utilization_ratio * 18.0 + rng.normal(0, 1.5, size=days), 2)
    packet_loss_pct = np.round(
        np.where(utilization_ratio > 0.8, (utilization_ratio - 0.8) * 9.0, rng.uniform(0.0, 0.04, size=days)),
        3,
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d"),
            "bandwidth_mbps": np.round(bandwidth, 2),
            "active_users": active_users,
            "latency_ms": latency_ms,
            "packet_loss_pct": packet_loss_pct,
            "allocated_capacity_mbps": link_capacity_mbps,
            "data_source": f"SYNTHETIC_DATACENTER_DAILY_SEED_{seed}",
        }
    )

    df.loc[[45, 112], "bandwidth_mbps"] = np.nan
    df.to_csv(filename, index=False)
    return df


def main():
    datasets_dir = Path(__file__).resolve().parent
    datasets_dir.mkdir(parents=True, exist_ok=True)

    campus_file = datasets_dir / "campus_core_hourly_60d.csv"
    isp_file = datasets_dir / "isp_edge_hourly_90d.csv"
    dc_file = datasets_dir / "datacenter_wan_daily_180d.csv"

    df_campus = generate_hourly_dataset(
        str(campus_file),
        days=60,
        base_mbps=430.0,
        growth_mbps=175.0,
        link_capacity_mbps=1000.0,
        profile="campus",
        seed=42,
        inject_missing_rate=0.01,
    )
    df_isp = generate_hourly_dataset(
        str(isp_file),
        days=90,
        base_mbps=1150.0,
        growth_mbps=520.0,
        link_capacity_mbps=2500.0,
        profile="isp",
        seed=77,
        inject_missing_rate=0.008,
    )
    df_dc = generate_daily_dataset(
        str(dc_file),
        days=180,
        base_mbps=2200.0,
        growth_mbps=1550.0,
        link_capacity_mbps=5000.0,
        seed=108,
    )

    print(f"Generated {campus_file.name}: {len(df_campus)} rows")
    print(f"Generated {isp_file.name}: {len(df_isp)} rows")
    print(f"Generated {dc_file.name}: {len(df_dc)} rows")


if __name__ == "__main__":
    main()

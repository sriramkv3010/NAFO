"""
src/network/channel_model.py
-----------------------------
3GPP TR 38.901 Urban Macro (UMa) Channel Model
Citation: 3GPP TR 38.901 V17.0.0 (2022)

This is the physics layer of the 5G digital twin.
Every SINR value in NAFO traces back to a specific
equation in this document — that is what makes the
simulation IEEE-defensible vs random.uniform().

Implements:
    - UMa LOS path loss (Table 7.4.1-1)
    - Log-normal shadowing (Section 7.4.1)
    - Rayleigh fast fading (Section 7.5)
    - Shannon channel capacity
    - Doppler shift for Hospital D mobility

Hospital distance profiles (distance to nearest gNB):
    Hospital A (tabular/EHR)   : 100m  — close to hospital edge node
    Hospital B (ECG)           : 500m  — moderate urban distance
    Hospital C (X-ray)         : 2000m — large hospital building, congested
    Hospital D (wearable/PPG)  : mobile, 200-800m, subject to handoff

Frequency band: 3.5 GHz (n78) — primary 5G NR mid-band
Bandwidth:
    URLLC (B, D): 20 MHz
    eMBB  (C)   : 100 MHz
    mMTC  (A)   : 5 MHz
"""

import numpy as np
from typing import Dict

# ── Physical constants and system parameters ──────────────────────────────────
FREQ_GHZ = 3.5  # 5G NR n78 band (GHz)
TX_POWER_DBM = 46.0  # gNB EIRP (dBm) — 5G NR macro cell
# 3GPP TS 38.104 Table 6.2.1-1: 46 dBm typical macro
SPEED_OF_LIGHT = 3e8  # m/s
NOISE_FIGURE_DB = 7.0  # UE receiver noise figure (dB) — 3GPP TS 38.101-1
BOLTZMANN_DBM = -174.0  # thermal noise density (dBm/Hz) at 290K

# Bandwidth per slice (MHz) — 3GPP TS 38.104 Table 5.3.5
SLICE_BANDWIDTH_MHZ = {
    "mMTC": 1.4,  # LTE-M (eMTC) — 3GPP TS 36.521-1
    # NB-IoT=0.18MHz, LTE-M=1.4MHz
    # max throughput ~1Mbps, correct for EHR tabular sync
    "URLLC": 20.0,  # Hospital B, D — real-time ECG/PPG
    "eMBB": 100.0,  # Hospital C — large DICOM imaging
}

# Noise power per slice: N = kTB + NF
# N (dBm) = -174 dBm/Hz + 10*log10(B_Hz) + NoiseFigure_dB
# 3GPP TS 38.101-1, Section 7.3
NOISE_DBM_PER_SLICE = {
    sl: BOLTZMANN_DBM + 10.0 * np.log10(bw * 1e6) + NOISE_FIGURE_DB
    for sl, bw in SLICE_BANDWIDTH_MHZ.items()
}
# mMTC  (1.4 MHz) → ≈ -106 dBm
# URLLC (20 MHz)  → ≈  -94 dBm
# eMBB  (100MHz)  → ≈  -87 dBm

# Operational SINR bounds — realistic cellular range
# Below -20dB: link failure; above 40dB: unrealistic (pilot contamination)
SINR_MIN_DB = -20.0
SINR_MAX_DB = 40.0

# Hospital network profiles
# distance_m: nominal distance to serving gNB
# shadow_std_db: log-normal shadowing standard deviation (TR 38.901 Table 7.4.1-1)
HOSPITAL_PROFILES = {
    "hospital_a": {
        "distance_m": 100.0,
        "slice": "mMTC",
        "shadow_std_db": 4.0,  # UMa LOS
        "is_mobile": False,
        "indoor_loss_db": 20.0,  # TR 38.901 Table 7.4.3-1 indoor office
        # EHR terminal inside hospital building
    },
    "hospital_b": {
        "distance_m": 500.0,
        "slice": "URLLC",
        "shadow_std_db": 6.0,  # UMa NLOS
        "is_mobile": False,
        "indoor_loss_db": 0.0,
    },
    "hospital_c": {
        "distance_m": 500.0,  # served by dedicated hospital DAS/small cell
        "slice": "eMBB",
        "shadow_std_db": 7.82,  # UMa NLOS high sigma
        "is_mobile": False,
        "indoor_loss_db": 0.0,  # indoor 5G DAS — no penetration loss
    },
    "hospital_d": {
        "distance_m": 300.0,
        "slice": "URLLC",
        "shadow_std_db": 6.0,
        "is_mobile": True,
        "indoor_loss_db": 0.0,  # outdoor wearable patient
    },
}


class TR38901ChannelModel:
    """
    3GPP TR 38.901 Urban Macro channel model.

    Computes per-round SINR and Shannon capacity for each hospital.
    All equations are numbered by table/section in TR 38.901 V17.0.0.

    Usage:
        model = TR38901ChannelModel(seed=42)
        sinr_db, capacity_mbps = model.compute(hospital_id, distance_m)
    """

    def __init__(self, freq_ghz: float = FREQ_GHZ, seed: int = 42):
        self.freq_ghz = freq_ghz
        self.rng = np.random.default_rng(seed)

    # ── Path Loss ─────────────────────────────────────────────────────────────

    def path_loss_uma_los(self, distance_m: float) -> float:
        """
        UMa LOS path loss.
        3GPP TR 38.901 V17.0.0, Table 7.4.1-1, Row 1 (UMa LOS).

        PL = 28.0 + 22*log10(d3D) + 20*log10(fc)
        Valid for: 10m <= d3D <= d_BP'

        Args:
            distance_m: 3D distance to gNB (metres)

        Returns:
            path loss in dB
        """
        d = max(distance_m, 10.0)  # minimum distance guard
        PL = 28.0 + 22.0 * np.log10(d) + 20.0 * np.log10(self.freq_ghz)
        return float(PL)

    def path_loss_uma_nlos(self, distance_m: float) -> float:
        """
        UMa NLOS path loss.
        3GPP TR 38.901 V17.0.0, Table 7.4.1-1, Row 2 (UMa NLOS).

        PL = 13.54 + 39.08*log10(d3D) + 20*log10(fc) - 0.6*(h_UT - 1.5)
        h_UT assumed 1.5m (pedestrian/wearable height).

        Args:
            distance_m: 3D distance to gNB (metres)

        Returns:
            path loss in dB
        """
        d = max(distance_m, 10.0)
        h_ut = 1.5  # UE height (metres) — pedestrian
        PL = (
            13.54
            + 39.08 * np.log10(d)
            + 20.0 * np.log10(self.freq_ghz)
            - 0.6 * (h_ut - 1.5)
        )
        return float(PL)

    # ── Shadowing ─────────────────────────────────────────────────────────────

    def shadowing_db(self, sigma_db: float) -> float:
        """
        Log-normal shadowing (large-scale fading).
        3GPP TR 38.901 V17.0.0, Section 7.4.1.

        X ~ N(0, sigma^2)
        sigma = 4 dB (LOS), 6-7.82 dB (NLOS) per Table 7.4.1-1.

        Args:
            sigma_db: standard deviation in dB

        Returns:
            shadowing value in dB (zero-mean)
        """
        return float(self.rng.normal(0.0, sigma_db))

    # ── Fast Fading ───────────────────────────────────────────────────────────

    def fast_fading_db(self, is_los: bool = False) -> float:
        """
        Small-scale (fast) fading.
        3GPP TR 38.901 V17.0.0, Section 7.5.

        LOS : Rician fading (dominant component + scatter)
        NLOS: Rayleigh fading (no dominant component)

        For Rayleigh: h = (N(0,1) + jN(0,1)) / sqrt(2)
        |h|^2 ~ Exponential(1) → power in dB = 20*log10(|h|)

        Returns:
            fast fading power in dB
        """
        if is_los:
            # Rician K=1: weak LOS component, less variance
            k_factor = 1.0
            h_los = np.sqrt(k_factor / (k_factor + 1))
            h_scat = self.rng.normal(0, 1 / np.sqrt(2 * (k_factor + 1)), 2)
            h = h_los + h_scat[0] + 1j * h_scat[1]
        else:
            # Rayleigh: no LOS component
            h = self.rng.normal(0, 1 / np.sqrt(2)) + 1j * self.rng.normal(
                0, 1 / np.sqrt(2)
            )

        fading_db = 20.0 * np.log10(np.abs(h) + 1e-10)
        return float(fading_db)

    # ── Doppler (Hospital D mobility) ─────────────────────────────────────────

    def doppler_shift_hz(
        self,
        velocity_mps: float = 1.4,  # walking speed (m/s)
        angle_deg: float = 45.0,  # angle to gNB
    ) -> float:
        """
        Doppler frequency shift due to UE mobility.
        3GPP TR 38.901 V17.0.0, Section 7.6.6.

        f_D = (v / c) * f_c * cos(theta)

        Args:
            velocity_mps: UE velocity in m/s (1.4 m/s = walking)
            angle_deg   : angle between velocity vector and gNB direction

        Returns:
            Doppler shift in Hz
        """
        f_c = self.freq_ghz * 1e9  # Hz
        theta = np.radians(angle_deg)
        f_D = (velocity_mps / SPEED_OF_LIGHT) * f_c * np.cos(theta)
        return float(f_D)

    def doppler_sinr_penalty_db(self, velocity_mps: float = 1.4) -> float:
        """
        Approximate SINR degradation due to Doppler spread.
        Higher velocity → more inter-carrier interference → SINR penalty.
        Empirical model: penalty ≈ 0.5 * log10(1 + f_D / subcarrier_spacing)
        OFDM subcarrier spacing: 30 kHz (5G NR numerology 1)

        Returns:
            SINR penalty in dB (negative)
        """
        f_D = abs(self.doppler_shift_hz(velocity_mps))
        subcarrier_spacing = 30e3  # Hz, 5G NR numerology 1
        penalty = -0.5 * np.log10(1.0 + f_D / subcarrier_spacing)
        return float(penalty)

    # ── SINR ──────────────────────────────────────────────────────────────────

    def compute_sinr_db(
        self,
        hospital_id: str,
        distance_m: float,
        is_los: bool = False,
        velocity_mps: float = 0.0,
    ) -> float:
        """
        Compute received SINR for one hospital in one round.

        SINR (dB) = Tx_power - Path_loss - Shadowing - Fast_fading
                    - Doppler_penalty - Noise_power(slice)

        Noise power is computed per slice bandwidth:
            N = kTB + NoiseFigure  (3GPP TS 38.101-1, Section 7.3)

        SINR is clipped to [-20, 40] dB — operational cellular range.
        Below -20dB the link fails; above 40dB is physically unrealistic.

        Args:
            hospital_id  : hospital identifier (for slice and noise lookup)
            distance_m   : distance to serving gNB
            is_los       : line-of-sight flag
            velocity_mps : UE velocity (non-zero for Hospital D)

        Returns:
            SINR in dB, clipped to [SINR_MIN_DB, SINR_MAX_DB]
        """
        profile = HOSPITAL_PROFILES[hospital_id]
        sigma = profile["shadow_std_db"]
        slice_type = profile["slice"]
        indoor_loss = profile.get("indoor_loss_db", 0.0)

        # Per-slice thermal noise power (dBm)
        noise_dbm = NOISE_DBM_PER_SLICE[slice_type]

        # Select path loss model
        if is_los:
            pl = self.path_loss_uma_los(distance_m)
        else:
            pl = self.path_loss_uma_nlos(distance_m)

        # Received power (dBm) — includes indoor penetration loss
        # TR 38.901 Table 7.4.3-1: indoor office 20dB, partial indoor 10dB
        rx_power = (
            TX_POWER_DBM
            - pl
            - indoor_loss
            - self.shadowing_db(sigma)
            - abs(self.fast_fading_db(is_los))
        )

        # Doppler penalty for mobile nodes
        if velocity_mps > 0:
            rx_power += self.doppler_sinr_penalty_db(velocity_mps)

        # SINR = received power - noise power (both in dBm)
        sinr_db = rx_power - noise_dbm

        # Clip to physically realistic operational range
        # Below -20dB: link failure / outage
        # Above  40dB: unrealistic for macro cell (pilot contamination limit)
        sinr_db = float(np.clip(sinr_db, SINR_MIN_DB, SINR_MAX_DB))
        return sinr_db

    # ── Shannon Capacity ──────────────────────────────────────────────────────

    def shannon_capacity_mbps(
        self,
        sinr_db: float,
        bandwidth_mhz: float,
    ) -> float:
        """
        Shannon-Hartley channel capacity.
        C = B * log2(1 + SINR_linear)

        CRITICAL: B must be in Hz for result to be in bits/s.
        Divide by 1e6 to get Mbps.

        3GPP TS 38.306 Table 4.1.2-1 defines maximum spectral efficiency
        at 8.0 bits/s/Hz (256-QAM, rate 948/1024). We apply this as a
        practical ceiling — Shannon is a theoretical limit.

        Args:
            sinr_db      : SINR in dB (should be pre-clipped)
            bandwidth_mhz: channel bandwidth in MHz

        Returns:
            capacity in Mbps (Shannon upper bound)
        """
        # Clip SINR before conversion to prevent extreme linear values
        sinr_db_clipped = float(np.clip(sinr_db, SINR_MIN_DB, SINR_MAX_DB))
        sinr_linear = 10.0 ** (sinr_db_clipped / 10.0)

        # Convert bandwidth to Hz for correct units
        bandwidth_hz = bandwidth_mhz * 1e6

        # Shannon capacity in bits/s, convert to Mbps
        capacity_bps = bandwidth_hz * np.log2(1.0 + sinr_linear)
        capacity_mbps = capacity_bps / 1e6

        # Practical ceiling: 3GPP max spectral efficiency 8.0 bits/s/Hz
        max_capacity_mbps = 8.0 * bandwidth_mhz

        # Hard cap for mMTC: LTE-M maximum is 1 Mbps DL
        # 3GPP TS 36.306: LTE-M peak data rate = 1 Mbps
        if bandwidth_mhz <= 1.4:
            max_capacity_mbps = min(max_capacity_mbps, 1.0)

        capacity_mbps = min(capacity_mbps, max_capacity_mbps)
        return float(capacity_mbps)

    # ── Full compute ──────────────────────────────────────────────────────────

    def compute(
        self,
        hospital_id: str,
        distance_m: float = None,
        is_los: bool = False,
        velocity_mps: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute all channel metrics for one hospital in one round.

        Args:
            hospital_id  : one of hospital_a/b/c/d
            distance_m   : override nominal distance (optional)
            is_los       : line-of-sight flag
            velocity_mps : UE mobility speed

        Returns:
            dict with sinr_db, capacity_mbps, path_loss_db, bandwidth_mhz
        """
        profile = HOSPITAL_PROFILES[hospital_id]
        d = distance_m if distance_m is not None else profile["distance_m"]
        slice_type = profile["slice"]
        bw_mhz = SLICE_BANDWIDTH_MHZ[slice_type]

        sinr_db = self.compute_sinr_db(
            hospital_id=hospital_id,
            distance_m=d,
            is_los=is_los,
            velocity_mps=velocity_mps,
        )
        capacity_mbps = self.shannon_capacity_mbps(sinr_db, bw_mhz)

        if is_los:
            pl_db = self.path_loss_uma_los(d)
        else:
            pl_db = self.path_loss_uma_nlos(d)

        return {
            "hospital": hospital_id,
            "slice": slice_type,
            "distance_m": d,
            "sinr_db": sinr_db,
            "capacity_mbps": capacity_mbps,
            "path_loss_db": pl_db,
            "bandwidth_mhz": bw_mhz,
        }

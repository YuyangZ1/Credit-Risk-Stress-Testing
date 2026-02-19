# Credit Risk Stress Testing (Basel Vasicek Framework)

A Python-based credit risk stress testing project implementing PD estimation, Basel Vasicek stress mapping, and facility-level Unexpected Loss (UL) analysis with visualization.

---

## Project Overview

This project performs end-to-end credit risk analysis using obligor-level Probability of Default (PD) and facility-level Exposure at Default (EAD) data.

Main components include:

- PIT PD aggregation by rating
- TTC PD construction
- Basel 99.9% stressed PD (Vasicek single-factor model)
- Facility-level Unexpected Loss (UL) computation
- Rating-level EAD aggregation
- Stress visualization (time series, histograms, 3D surfaces)

---

## Methodology

### TTC PD

Time-averaged PD per obligor:
PD_TTC = average(PD over time)

### Basel Stressed PD (Vasicek Mapping)

PD_stress = Φ((Φ⁻¹(PD_TTC) + √ρ · Φ⁻¹(0.999)) / √(1−ρ))

Where:

- ρ = asset correlation
- 0.999 = Basel confidence level
- Φ = standard normal CDF

### Unexpected Loss (Facility Level)

UL = EAD × LGD^DT × (PD_stress − PD_TTC)

LGD^DT is estimated from default events during stress periods (GFC / COVID).

---

## Tech Stack

- Python
- pandas
- numpy
- scipy
- matplotlib

---

## How to Run

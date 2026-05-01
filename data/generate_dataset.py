"""
Generate a realistic crop recommendation dataset.
Based on UCI Crop Recommendation Dataset structure.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# Crop profiles: (N_mean, P_mean, K_mean, temp_mean, humidity_mean, ph_mean, rainfall_mean)
crop_profiles = {
    "rice":        (80, 45, 40, 23, 82, 6.5, 200),
    "maize":       (80, 42, 42, 22, 65, 6.2, 60),
    "chickpea":    (40, 68, 80, 18, 16, 7.5, 80),
    "kidneybeans": (20, 67, 20, 19, 21, 5.7, 105),
    "pigeonpeas":  (20, 67, 20, 27, 48, 5.7, 148),
    "mothbeans":   (21, 48, 20, 28, 52, 6.9, 51),
    "mungbean":    (20, 48, 20, 28, 85, 6.7, 48),
    "blackgram":   (40, 67, 19, 29, 64, 7.0, 68),
    "lentil":      (18, 68, 19, 24, 65, 6.9, 46),
    "pomegranate": (18, 18, 40, 21, 90, 6.3, 107),
    "banana":      (100,82, 50, 27, 80, 6.0, 105),
    "mango":       (20, 27, 30, 31, 50, 5.8, 95),
    "grapes":      (23, 132,200, 24, 81, 6.0, 70),
    "watermelon":  (99, 17, 50, 25, 85, 6.5, 50),
    "muskmelon":   (100,17, 50, 28, 92, 6.4, 25),
    "apple":       (21, 134,200, 21, 92, 5.9, 113),
    "orange":      (20, 16, 10, 22, 92, 7.0, 110),
    "papaya":      (49, 59, 50, 33, 92, 6.7, 143),
    "coconut":     (22, 16, 30, 27, 94, 6.0, 176),
    "cotton":      (118,46, 43, 24, 79, 6.8, 80),
    "jute":        (78, 46, 40, 24, 80, 6.5, 175),
    "coffee":      (101,28, 29, 25, 58, 6.8, 158),
}

samples_per_crop = 100
records = []

for crop, (N, P, K, T, H, ph, R) in crop_profiles.items():
    for _ in range(samples_per_crop):
        records.append({
            "N":         max(0, np.random.normal(N, N * 0.12)),
            "P":         max(0, np.random.normal(P, P * 0.12)),
            "K":         max(0, np.random.normal(K, K * 0.12)),
            "temperature": np.random.normal(T, 2.0),
            "humidity":  np.clip(np.random.normal(H, H * 0.08), 10, 100),
            "ph":        np.clip(np.random.normal(ph, 0.4), 3.5, 9.5),
            "rainfall":  max(0, np.random.normal(R, R * 0.15)),
            "label":     crop,
        })

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("/home/claude/crop_recommendation/data/crop_data.csv", index=False)
print(f"Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
print(df["label"].value_counts())

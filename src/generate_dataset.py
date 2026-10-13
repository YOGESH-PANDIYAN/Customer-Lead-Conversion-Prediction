
import numpy as np
import pandas as pd

# Reproducible random generator
rng = np.random.default_rng(42)

n = 2000

data = pd.DataFrame({
    "lead_id": np.arange(1, n + 1),
    "age": rng.integers(22, 65, n),
    "location": rng.choice(
        ["Chennai", "Madurai", "Coimbatore"],
        n
    ),
    "budget": rng.integers(
        2000000, 15000001, n
    ),
    "property_type": rng.choice(
        ["Apartment", "Villa", "Plot"],
        n
    ),
    "lead_source": rng.choice(
        ["Website", "Google", "Facebook"],
        n
    ),
    "calls": rng.integers(0, 7, n),
    "follow_ups": rng.integers(0, 9, n),
    "site_visit": rng.choice(
        [0, 1], n, p=[0.6, 0.4]
    ),
    "response_time_hours": rng.integers(
        1, 49, n
    )
})

# Simulate varied conversion outcomes
score = (
    -1.8
    + 0.22 * data["calls"]
    + 0.15 * data["follow_ups"]
    + 0.85 * data["site_visit"]
    - 0.025 * data["response_time_hours"]
    + 0.25 * (data["lead_source"] == "Google")
    + 0.15 * (data["budget"] >= 7000000)
)

# Convert score into probabilities
probability = 1 / (1 + np.exp(-score))

# Randomly sample conversion outcomes
data["converted"] = rng.binomial(1, probability)

# Save dataset
data.to_csv("data/leads_large.csv", index=False)

print("Dataset generated successfully!")
print("Dataset shape:", data.shape)
print("\nConversion counts:")
print(data["converted"].value_counts())
print("\nConversion rate:")
print(f"{data['converted'].mean() * 100:.2f}%")

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("data/leads.csv")

# -----------------------------
# Basic Information
# -----------------------------

print("Dataset Shape:")
print(data.shape)

print("\nDataset Columns:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())

# -----------------------------
# Conversion Analysis
# -----------------------------

conversion_rate = data["converted"].mean() * 100

print("\nConversion Rate:")
print(f"{conversion_rate:.2f}%")

print("\nConversion Count:")
print(data["converted"].value_counts())

# -----------------------------
# Lead Source Analysis
# -----------------------------

print("\nLeads by Source:")
print(data["lead_source"].value_counts())

source_conversion = data.groupby("lead_source")["converted"].mean() * 100

print("\nConversion Rate by Source:")
print(source_conversion)

# -----------------------------
# Site Visit Analysis
# -----------------------------

site_visit_conversion = data.groupby("site_visit")["converted"].mean() * 100

print("\nConversion Rate by Site Visit:")
print(site_visit_conversion)

# -----------------------------
# Follow-up Analysis
# -----------------------------

followup_conversion = data.groupby("follow_ups")["converted"].mean() * 100

print("\nConversion Rate by Follow-ups:")
print(followup_conversion)

# -----------------------------
# Call Analysis
# -----------------------------

call_conversion = data.groupby("calls")["converted"].mean() * 100

print("\nConversion Rate by Calls:")
print(call_conversion)

# -----------------------------
# Budget Analysis
# -----------------------------

budget_conversion = data.groupby("converted")["budget"].mean()

print("\nAverage Budget:")
print(budget_conversion)

# -----------------------------
# Visualization 1
# Conversion Rate by Lead Source
# -----------------------------

source_conversion.plot(kind="bar")

plt.title("Conversion Rate by Lead Source")
plt.xlabel("Lead Source")
plt.ylabel("Conversion Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("visualizations/conversion_by_source.png")

plt.show()

# -----------------------------
# Visualization 2
# Conversion Rate by Site Visit
# -----------------------------

site_visit_conversion.plot(kind="bar")

plt.title("Conversion Rate by Site Visit")
plt.xlabel("Site Visit")
plt.ylabel("Conversion Rate (%)")
plt.xticks([0, 1], ["No", "Yes"], rotation=0)

plt.tight_layout()

plt.savefig("visualizations/conversion_by_site_visit.png")

plt.show()

# -----------------------------
# Visualization 3
# Conversion Rate by Follow-ups
# -----------------------------

followup_conversion.plot(kind="bar")

plt.title("Conversion Rate by Number of Follow-ups")
plt.xlabel("Number of Follow-ups")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig("visualizations/conversion_by_followups.png")

plt.show()

# -----------------------------
# Visualization 4
# Conversion Rate by Calls
# -----------------------------

call_conversion.plot(kind="bar")

plt.title("Conversion Rate by Number of Calls")
plt.xlabel("Number of Calls")
plt.ylabel("Conversion Rate (%)")

plt.tight_layout()

plt.savefig("visualizations/conversion_by_calls.png")

plt.show()
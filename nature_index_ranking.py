# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 14:21:00 2026

@author: Jyotirmoy Paul
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load institution share data
# -------------------------------
all_subject_data = pd.read_csv("institution-2025.csv")

# Ensure shares are numeric
all_subject_data.iloc[:, 3] = pd.to_numeric(all_subject_data.iloc[:, 3], errors="coerce")

country_share = (
    all_subject_data
    .groupby(all_subject_data.columns[2])[all_subject_data.columns[3]]
    .sum()
    .sort_values(ascending=False)
)

country_share_df = country_share.reset_index()
country_share_df.columns = ["Country", "Total_Shares"]

# -------------------------------
# Load world population data
# -------------------------------
world_population = pd.read_excel("population.xlsx", header=None)
world_population = world_population.iloc[:, :2]
world_population.columns = ["Country", "Population"]

# -------------------------------
# CLEAN COUNTRY NAMES
# -------------------------------
def clean_country(col):
    return (
        col.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
        .str.lower()
    )

country_share_df["Country"] = clean_country(country_share_df["Country"])
world_population["Country"] = clean_country(world_population["Country"])

# Fix common naming differences
replacements = {
    "united states of america (usa)": "united states",
    "united kingdom (uk)": "united kingdom",
    "south korea": "korea, republic of",
    "russia": "russian federation",
    "iran": "iran, islamic republic of",
    "venezuela": "venezuela, bolivarian republic of"
}

country_share_df["Country"] = country_share_df["Country"].replace(replacements)

# -------------------------------
# MERGE DATASETS
# -------------------------------
merged = country_share_df.merge(world_population, on="Country", how="inner")

# -------------------------------
# CALCULATE SHARES PER CAPITA
# -------------------------------
merged["Shares_per_Capita"] = merged["Total_Shares"] / merged["Population"]

share_per_capita = merged.sort_values("Shares_per_Capita", ascending=False)

# -------------------------------
# PLOT TOP N BY TOTAL SHARES
# -------------------------------
top_n = 30

top_countries = (
    country_share_df
    .sort_values("Total_Shares", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)

# Rank labels
ranked_labels = [
    f"{i+1}. {c.title()}"
    for i, c in enumerate(top_countries["Country"])
]

plt.figure(figsize=(10, 0.5 * top_n + 2))

colors = plt.cm.plasma(np.linspace(0, 1, len(top_countries)))

plt.barh(ranked_labels, top_countries["Total_Shares"], color=colors)

plt.xlabel("Total Shares")
plt.title(f"Top {top_n} Countries by Total Shares")
plt.gca().invert_yaxis()

# Value labels
for i, value in enumerate(top_countries["Total_Shares"]):
    plt.text(value, i, f" {int(value):,}", va="center")

plt.tight_layout()
plt.savefig("total_share.png", dpi=300)
plt.show()

#%% per capita 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load institution share data
# -------------------------------
all_subject_data = pd.read_csv('institution-2025.csv')

country_share = (
    all_subject_data
        .groupby(all_subject_data.columns[2])[all_subject_data.columns[3]]
        .sum()
        .sort_values(ascending=False)
)

country_share_df = country_share.reset_index()
country_share_df.columns = ["Country", "Total_Shares"]

# -------------------------------
# Load world population data
# -------------------------------
world_population = pd.read_excel('population.xlsx', header=None)
world_population = world_population.iloc[:, :2]
world_population.columns = ["Country", "Population"]

# -------------------------------
# CLEAN COUNTRY NAMES
# -------------------------------

# Remove weird spaces and normalize
country_share_df["Country"] = (
    country_share_df["Country"]
    .str.replace('\xa0', '', regex=False)
    .str.strip()
    .str.lower()
)

world_population["Country"] = (
    world_population["Country"]
    .str.replace('\xa0', '', regex=False)
    .str.strip()
    .str.lower()
)

# Fix common naming differences
replacements = {
    "united states of america (usa)": "united states",
    "united kingdom (uk)": "united kingdom",
    "south korea": "korea, republic of",
    "russia": "russian federation",
    "iran": "iran, islamic republic of",
    "venezuela": "venezuela, bolivarian republic of"
}

country_share_df["Country"] = country_share_df["Country"].replace(replacements)

# -------------------------------
# MERGE DATASETS
# -------------------------------
merged = country_share_df.merge(world_population, on="Country", how="inner")

# -------------------------------
# CALCULATE SHARES PER CAPITA
# -------------------------------
merged["Shares_per_Capita"] = 1e6* merged["Total_Shares"] / merged["Population"]

share_per_capita = merged.sort_values("Shares_per_Capita", ascending=False)

print(share_per_capita[["Country", "Total_Shares", "Population", "Shares_per_Capita"]])

# -------------------------------
# OPTIONAL: Plot Top N Countries
# -------------------------------
top_n = 40
top_data = share_per_capita.head(top_n)

plt.figure(figsize=(10, 0.5 * top_n + 2))
colors = plt.cm.plasma(np.linspace(0, 1, len(top_data)))

# Add rank numbers
ranks = np.arange(1, len(top_data) + 1)
labels = [f"{rank}. {country}" for rank, country in zip(ranks, top_data["Country"])]

plt.barh(labels, top_data["Shares_per_Capita"], color=colors)

plt.xlabel("Shares per Capita")
plt.title(f"Top {top_n} Countries by Shares per Capita")

plt.gca().invert_yaxis()

# Remove bounding box (spines)
ax = plt.gca()
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

# Value labels
for i, value in enumerate(top_data["Shares_per_Capita"]):
    plt.text(value, i, f' {value:.2e}', va='center')



plt.tight_layout()
plt.savefig('percapita.png', dpi=300)
plt.show()

#%% get rank 
# -------------------------------
# ADD RANKING COLUMN
# -------------------------------
share_per_capita = share_per_capita.reset_index(drop=True)
share_per_capita["Rank"] = share_per_capita.index + 1

# -------------------------------
# COUNTRY NAME ALIASES
# -------------------------------
aliases = {
    "usa": "united states",
    "uk": "united kingdom",
    "south korea": "korea, republic of",
    "russia": "russian federation",
    "iran": "iran, islamic republic of",
    "venezuela": "venezuela, bolivarian republic of"
}

# -------------------------------
# LOOKUP FUNCTION
# -------------------------------
def get_country_rank(country_name):
    country_key = country_name.strip().lower()
    country_key = aliases.get(country_key, country_key)

    match = share_per_capita[share_per_capita["Country"] == country_key]

    if not match.empty:
        rank = int(match["Rank"].values[0])
        shares = int(match["Total_Shares"].values[0])
        population = int(match["Population"].values[0])
        shares_pc = match["Shares_per_Capita"].values[0]

        print(f"\n📊 Country: {country_key.title()}")
        print(f"🏅 Rank: #{rank}")
        print(f"📦 Total Shares: {shares:,}")
        print(f"👥 Population: {population:,}")
        print(f"⚖️ Shares per Capita: {shares_pc:.2e}\n")
    else:
        print("\n❌ Country not found in dataset. Check spelling.\n")

# -------------------------------
# INTERACTIVE USER INPUT (Spyder)
# -------------------------------
while True:
    user_input = input("Enter a country name to find its rank (or type 'exit' to quit): ")
    
    if user_input.strip().lower() == "exit":
        print("Done 👍")
        break

    get_country_rank(user_input)

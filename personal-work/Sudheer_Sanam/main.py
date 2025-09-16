from pathlib import Path
from src.data_preprocessing import clean_and_merge_data
from src.data_merge_scoring import load_clean_data, merge_all_data, develop_weather_sensitivity_score

PROCESSED_PATH = Path("data/processed")
INTERIM_PATH = Path("data/interim")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

weather_df, charger_df, session_df, merged_charger_sessions_df = clean_and_merge_data()

# Save processed files
weather_df.to_csv(PROCESSED_PATH / "weather_data_clean.csv", index=False)
charger_df.to_csv(PROCESSED_PATH / "charger_data_clean.csv", index=False)
session_df.to_csv(PROCESSED_PATH / "session_data_clean.csv", index=False)
merged_charger_sessions_df.to_csv(PROCESSED_PATH / "merged_charger_sessions_clean.csv", index=False)

print("Data processing complete and files saved to data/processed/")
print(merged_charger_sessions_df.head())

INTERIM_PATH.mkdir(parents=True, exist_ok=True)

weather, stations, sessions, merged_sessions = load_clean_data()
combined = merge_all_data(weather, merged_sessions)

# Apply scoring logic
final_dataset = develop_weather_sensitivity_score(combined)

# Save intermediate merged data
combined.to_csv(INTERIM_PATH / "merged_dataset_interim.csv", index=False)

# Save final dataset with weather sensitivity scoring
final_dataset.to_csv(INTERIM_PATH / "final_ev_weather_sensitive_routes.csv", index=False)


print("Data merging and scoring completed. Check 'data/interim/' folder.")

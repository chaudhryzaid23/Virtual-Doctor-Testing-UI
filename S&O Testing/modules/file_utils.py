import os
import re
import csv
import random
import string
from datetime import datetime
import pandas as pd

OUTPUT_DIR = "outputs"
LOG_FILE = "llm_log.csv"

# --- Filename Sanitizer ---
def clean_filename(text):
    text = re.sub(r'[^a-zA-Z0-9_\- ]', '', text)
    return text.replace(" ", "_")[:40]

# --- Save to CSV Log ---
def log_to_csv(data: dict, path: str = LOG_FILE):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def delete_last_rating(log_file_path: str) -> bool:
    if not os.path.exists(log_file_path):
        return False

    df = pd.read_csv(log_file_path)

    if df.empty:
        return False

    df = df.iloc[:-1]  # Drop the last row
    df.to_csv(log_file_path, index=False)

    return True

def save_output_file(content: str, filename: str, directory: str = OUTPUT_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def generate_custom_id() -> str:
    chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=40))
    return f"{chars[:12]}--{chars[12:24]}--{chars[24:]}" 
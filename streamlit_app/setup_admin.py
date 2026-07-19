"""
Run this script ONCE to create an admin account before using the app.
Usage: python setup_admin.py

Edit DB_CONFIG below to match your MySQL setup, or better, load it the
same way app.py does (from .streamlit/secrets.toml) so credentials stay
in one place.
"""

import database as db
import getpass

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "credit_app",
    "password": "Vishu@2006",       # <-- match your MySQL password
    "database": "credit_risk_db",
}

if __name__ == "__main__":
    print("=== Create a new admin account ===")
    username = input("Choose a username: ").strip()
    password = getpass.getpass("Choose a password (hidden as you type): ").strip()

    db.init_db(DB_CONFIG)  # make sure tables exist first
    db.create_admin(DB_CONFIG, username, password)
    print(f"\nAdmin account '{username}' created successfully.")
    print("You can now log in to the Streamlit app with these credentials.")

"""
Run this script ONCE to create an admin account in the Aiven Cloud Database.
Usage: python setup_admin.py
"""

import database as db
import getpass

# Ask you for the Aiven DB password right in the terminal so it's never saved in the file!
aiven_password = getpass.getpass("Paste your Aiven Cloud Password (hidden as you type): ").strip()

DB_CONFIG = {
    "host": "mysql-19f4630d-vishweshankam38-080c.g.aivencloud.com",
    "port": 24194,
    "user": "avnadmin",
    "password": aiven_password,  # <-- Uses the password you type dynamically
    "database": "defaultdb",
}

if __name__ == "__main__":
    print("\n=== Connecting to Aiven Cloud & Initializing Tables ===")
    db.init_db(DB_CONFIG)
    
    print("\n=== Create a new admin account for the Web App ===")
    username = input("Choose a login username: ").strip()
    password = getpass.getpass("Choose a login password (hidden as you type): ").strip()

    db.create_admin(DB_CONFIG, username, password)
    print(f"\nAdmin account '{username}' created successfully in Aiven Cloud.")
    print("This specific account will be authorized to access your public Streamlit Cloud app.")
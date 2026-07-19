"""
Database module: MySQL storage for prediction history + admin authentication.
"""

import mysql.connector
import bcrypt
from datetime import datetime


def get_connection(config):
    """Open a new MySQL connection using the given config dict."""
    return mysql.connector.connect(
        host=config["host"],
        port=config.get("port", 3306),
        user=config["user"],
        password=config["password"],
        database=config["database"],
    )


def init_db(config):
    """Create the required tables if they don't already exist."""
    conn = get_connection(config)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            prediction_timestamp DATETIME NOT NULL,
            age INT,
            monthly_income FLOAT,
            debt_ratio FLOAT,
            revolving_utilization FLOAT,
            open_credit_lines INT,
            real_estate_loans INT,
            late_30_59 INT,
            late_60_89 INT,
            late_90_plus INT,
            dependents INT,
            risk_probability FLOAT NOT NULL,
            risk_prediction TINYINT NOT NULL,
            predicted_by VARCHAR(100)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at DATETIME NOT NULL
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()


def create_admin(config, username, password):
    """Create a new admin user with a securely hashed password."""
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn = get_connection(config)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO admins (username, password_hash, created_at) VALUES (%s, %s, %s)",
        (username, password_hash, datetime.now())
    )
    conn.commit()
    cursor.close()
    conn.close()


def verify_admin(config, username, password):
    """Check username/password against stored hash. Returns True/False."""
    conn = get_connection(config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT password_hash FROM admins WHERE username = %s", (username,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        return False

    return bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8"))


def insert_prediction(config, input_dict, proba, prediction, predicted_by):
    """Store one prediction record with all inputs, result, and timestamp."""
    conn = get_connection(config)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            prediction_timestamp, age, monthly_income, debt_ratio, revolving_utilization,
            open_credit_lines, real_estate_loans, late_30_59, late_60_89, late_90_plus,
            dependents, risk_probability, risk_prediction, predicted_by
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        datetime.now(),
        input_dict["age"], input_dict["MonthlyIncome"], input_dict["DebtRatio"],
        input_dict["RevolvingUtilizationOfUnsecuredLines"],
        input_dict["NumberOfOpenCreditLinesAndLoans"], input_dict["NumberRealEstateLoansOrLines"],
        input_dict["NumberOfTime30-59DaysPastDueNotWorse"],
        input_dict["NumberOfTime60-89DaysPastDueNotWorse"],
        input_dict["NumberOfTimes90DaysLate"], input_dict["NumberOfDependents"],
        float(proba), int(prediction), predicted_by
    ))
    conn.commit()
    cursor.close()
    conn.close()


def get_all_predictions(config):
    """Fetch all stored predictions, most recent first."""
    conn = get_connection(config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM predictions ORDER BY prediction_timestamp DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def clear_predictions(config):
    """Delete all prediction records."""
    conn = get_connection(config)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    cursor.close()
    conn.close()

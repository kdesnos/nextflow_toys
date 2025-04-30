import os
import sqlite3
from trace_table_manager import TraceTableManager, TraceEntry


class NextflowTraceDBManager:
    def __init__(self, db_path):
        """
        Initialize the NextflowTraceDBManager with a connection to the specified SQLite database.

        :param db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.connection = None
        self.trace_manager = None

    def connect(self):
        """
        Establish a connection to the SQLite database.
        """
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            # Activate foreign_keys support on all connections as it is not persistent.
            self.connection.execute("PRAGMA foreign_keys = ON;")
            self.trace_manager = TraceTableManager(self.connection)

    def isConnected(self):
        """
        Check if the database connection is established.

        :return: True if the database is connected, False otherwise.
        """
        return self.connection is not None

    def isDatabaseEmpty(self):
        """
        Check if the connected SQLite database is empty.

        :return: True if the database is empty, False otherwise.
        :raises Exception: If the database is not connected.
        """
        if not self.isConnected():
            raise Exception("Database is not connected.")

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return len(tables) == 0

    def createTables(self, force=False):
        """
        Create tables in the SQLite database using the SQL script file.

        :param force: If True, delete the database file and recreate it.
        :raises Exception: If the database is not connected and force is False.
        """
        if not self.isConnected() and not force:
            raise Exception("Database is not connected.")

        if force:
            # Close the connection if it's open
            if self.connection:
                self.close()

            # Delete the database file
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                print(f"Database file '{self.db_path}' deleted.")

            # Reconnect to recreate the database
            self.connect()

        # Read and execute the SQL script to create tables
        with open(os.path.join(os.path.dirname(__file__), "nextflow_trace_DB_creation_script.sql"), "r") as sql_file:
            sql_script = sql_file.read()
            cursor = self.connection.cursor()
            cursor.executescript(sql_script)

        self.connection.commit()
        print("Tables created successfully.")

    def close(self):
        """
        Close the connection to the SQLite database.
        """
        if self.connection:
            self.connection.close()
            self.connection = None

    def getUserVersion(self):
        """
        Retrieve the user_version of the SQLite database.

        :return: The user_version of the database as an integer.
        :raises Exception: If the database is not connected.
        """
        if not self.isConnected():
            raise Exception("Database is not connected.")

        cursor = self.connection.cursor()
        cursor.execute("PRAGMA user_version;")
        version = cursor.fetchone()[0]
        return version


# Main prog
if __name__ == "__main__":
    # Initialize the database manager with the path to the SQLite database
    db_manager = NextflowTraceDBManager("nf_trace_db.sqlite")
    
    # Establish a connection to the database
    db_manager.connect()
    print("Connected to the database.")
    
    # Check if the database is empty and create tables if necessary
    if db_manager.isDatabaseEmpty():
        db_manager.createTables()
    else:
        db_manager.createTables(force=True)
    print("Tables created successfully.")
    
    # Retrieve and print the user_version of the database
    user_version = db_manager.getUserVersion()
    print(f"Database user_version: {user_version}")

    # Create a fake trace entry
    fake_trace1 = TraceEntry(tId=0, day="2023-01-02", name="sleepy_einstein")
    fake_trace2 = TraceEntry(tId=0, day="2025-01-02", name="funny_curie")
    print(f"Adding trace entry: {fake_trace1}")
    
    # Add the trace entry to the database
    db_manager.trace_manager.addTraceEntry(fake_trace1)
    db_manager.trace_manager.addTraceEntry(fake_trace2)
    print(f"Trace entry added successfully with id: {fake_trace1.tId}.")

    # Add multiple trace entries to the database
    trace_entries = [
        TraceEntry(tId=0, day="2023-03-01", name="happy_tesla"),
        TraceEntry(tId=0, day="2023-03-02", name="brilliant_newton"),
        TraceEntry(tId=0, day="2023-03-03", name="curious_darwin"),
    ]
    db_manager.trace_manager.addTraces(trace_entries)
    print("Multiple trace entries added successfully.")

    # Retrieve a specific trace entry by name
    trace = db_manager.trace_manager.getTraceEntry("sleepy_einstein")
    if trace:
        print(f"Retrieved trace entry: {trace}")
    else:
        print("Trace entry not found.")
    
    # Retrieve all trace entries
    all_traces = db_manager.trace_manager.getAllTraces()
    print("All trace entries:")
    for trace in all_traces:
        print(trace)

    # Remove a specific trace entry by name
    if db_manager.trace_manager.removeTraceEntry("sleepy_einstein"):
        print("Trace entry 'sleepy_einstein' removed successfully.")
    else:
        print("Trace entry 'sleepy_einstein' does not exist.")     

    db_manager.close()
    print("Connection closed.")
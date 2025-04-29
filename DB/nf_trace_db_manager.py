import os
import sqlite3

class NextflowTraceDBManager:
    def __init__(self, db_path):
        """
        Initialize the NextflowTraceDBManager with a connection to the specified SQLite database.

        :param db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.connection = None

    def connect(self):
        """
        Establish a connection to the SQLite database.
        """
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)

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

        :param force: If True, force execution of the SQL script even if the database is not empty.
        :raises Exception: If the database is not connected or if the database is not empty and force is False.
        """
        if not self.isConnected():
            raise Exception("Database is not connected.")

        if not self.isDatabaseEmpty() and not force:
            raise Exception("Database is not empty. Use force=True to overwrite.")

        cursor = self.connection.cursor()

        if force:
            # Drop all existing tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table_name, in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")

        # Read and execute the SQL script
        with open(os.path.join(os.path.dirname(__file__), "nextflow_trace_DB_creation_script.sql"), "r") as sql_file:
            sql_script = sql_file.read()
            cursor.executescript(sql_script)

        self.connection.commit()

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
        user_version = cursor.fetchone()[0]
        return user_version


    def addTraceEntry(self, trace_entry):
        """
        Add a TraceEntry instance to the Traces table in the SQLite database and update the tId of the input trace_entry.

        :param trace_entry: An instance of TraceEntry.
        :raises Exception: If the database is not connected.
        """
        if not self.isConnected():
            raise Exception("Database is not connected.")

        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO Traces (day, name) VALUES (?, ?);",
            (trace_entry.day, trace_entry.name),
        )
        self.connection.commit()

        # Retrieve the last inserted row ID and update the trace_entry's tId
        trace_entry.tId = cursor.lastrowid


class TraceEntry:
    def __init__(self, tId, day, name):
        """
        Initialize a TraceEntry instance.

        :param tId: Trace ID (integer).
        :param day: Day (string).
        :param name: Name (string).
        """
        self.tId = tId
        self.day = day
        self.name = name

    def __repr__(self):
        return f"TraceEntry(tId={self.tId}, day='{self.day}', name='{self.name}')"     


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
    print("Tables created successfully.")
    
    # Retrieve and print the user_version of the database
    user_version = db_manager.getUserVersion()
    print(f"Database user_version: {user_version}")

    # Create a fake trace entry
    fake_trace = TraceEntry(tId=0, day="2023-01-02", name="sleepy_einstein")
    print(f"Adding trace entry: {fake_trace}")
    
    # Add the trace entry to the database
    db_manager.addTraceEntry(fake_trace)
    print(f"Trace entry added successfully with id: {fake_trace.tId}.")

    db_manager.close()
    print("Connection closed.")
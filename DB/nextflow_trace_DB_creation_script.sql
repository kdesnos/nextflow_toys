-- Database version 
-- (to be changed when upgrading the structure)
PRAGMA user_version = 0;
-- Activate cascaded deletion (Not persistent in SQLite, should be done at every connection)
PRAGMA foreign_keys = ON;

-- Create Tables
-- Traces table makes it possible to manage multiple nextflow run
-- within a single database.
CREATE TABLE IF NOT EXISTS Traces (
	tId INTEGER PRIMARY KEY UNIQUE NOT NULL, -- Trace ID
	day TEXT,
	name TEXT UNIQUE NOT NULL
);

-- Process Table contains all process as defined in the nf files.
-- Aliased names are not contained in this table.
CREATE TABLE IF NOT EXISTS Processes (
    pId INTEGER PRIMARY KEY UNIQUE NOT NULL, -- Process ID
    name TEXT NOT NULL,
    path TEXT NOT NULL  -- Path of the nf files containing the process definition.
    					-- Should be relative to the project folder for portability of the DB.
);

-- Table containing all resolved names of Process (aliased and non aliased)
CREATE TABLE IF NOT EXISTS ResolvedProcessNames (
	rId INTEGER PRIMARY KEY UNIQUE NOT NULL, -- Resolved ID
	pId INTEGER REFERENCES Processes(pId) ON DELETE CASCADE,
	name TEXT NOT NULL UNIQUE
); 

-- Table containing the list of input and output parameters of processes
CREATE TABLE IF NOT EXISTS ProcessInputs (
	pId INTEGER REFERENCES Processes (pId) ON DELETE CASCADE,
	rank INTEGER NOT NULL,
	type TEXT,
	name TEXT NOT NULL,
	
	Constraint PK_ProcessInputs Primary Key (pId, rank)
);

-- ProcessExecutions table contain the useful information from the html report.
CREATE TABLE IF NOT EXISTS ProcessExecutions (
	eId Integer UNIQUE PRIMARY KEY NOT NULL, --Execution ID
	tId INTEGER REFERENCES Traces (tId) ON DELETE CASCADE,
	rId INTEGER REFERENCES ResolvedProcessNames (rId) ON DELETE CASCADE,
	instance INTEGER NOT NULL,
	hash TEXT NOT NULL,
	time Real NOT NULL, -- Execution time in milliseconds
	
	UNIQUE (tId, rId, instance, hash)
);

-- ProcessExecParams contains the parameter values associated to process
-- executions listed in the ProcessExecutions table
CREATE TABLE IF NOT EXISTS ProcessExecParams (
	eId INTEGER REFERENCES ProcessExecutions (eId) ON DELETE CASCADE,
	rank INTEGER NOT NULL,
	value TEXT NOT NULL,
	
	Constraint PK_ProcessExecParams Primary Key (eId, rank)
);
-- Clean
DROP TABLE IF EXISTS Traces;
DROP TABLE IF EXISTS Processes;
DROP TABLE IF EXISTS ProcessInputs;
DROP TABLE IF EXISTS ResolvedProcessNames;
DROP TABLE IF EXISTS ProcessExecutions;
DROP TABLE IF EXISTS ProcessExecParams;

-- Create Tables
CREATE TABLE IF NOT EXISTS Traces (
	tId INTEGER PRIMARY KEY UNIQUE, -- Trace ID
	day TEXT,
	name TEXT UNIQUE
);

-- Process Table contains all process as defined in the nf files.
-- Aliased names are not contained in this table.
CREATE TABLE IF NOT EXISTS Processes (
    pId INTEGER PRIMARY KEY UNIQUE, -- Process ID
    name TEXT NOT NULL,
    path TEXT NOT NULL  -- Path of the nf files containing the process definition.
    					-- Should be relative to the project folder for portability of the DB.
);

-- Table containing all resolved names of Process (aliased and non aliased)
CREATE TABLE IF NOT EXISTS ResolvedProcessNames (
	rId INTEGER PRIMARY KEY UNIQUE, -- Resolved ID
	pId INTEGER,
	name TEXT NOT NULL UNIQUE,
	
	FOREIGN KEY (pId) REFERENCES Processes (pId)
); 

-- Table containing the list of input and output parameters of processes
CREATE TABLE IF NOT EXISTS ProcessInputs (
	pId INTEGER,
	rank INTEGER,
	type TEXT,
	name TEXT,
	FOREIGN KEY (pId) REFERENCES Processes (pId),
	
	Constraint PK_ProcessInputs Primary Key (pId, rank)
);

CREATE TABLE IF NOT EXISTS ProcessExecutions (
	eId Integer UNIQUE PRIMARY KEY, --Execution ID
	tId INTEGER,
	rId INTEGER,
	instance INTEGER,
	hash TEXT,
	
	FOREIGN KEY (tId) REFERENCES Traces (tId),
	FOREIGN KEY (rId) REFERENCES ResolvedProcessNames (rId),
	UNIQUE (tId, rId, instance, hash)
);

CREATE TABLE IF NOT EXISTS ProcessExecParams (
	eId INTEGER,
	rank INTEGER,
	value TEXT,
	
	FOREIGN KEY (eId) REFERENCES ProcessExecutions (eId),
	Constraint PK_ProcessExecParams Primary Key (eId, rank)
);

-- Populate DB for tests
INSERT INTO Processes VALUES (1, "bidon", "/path/to/nf");
INSERT INTO ResolvedProcessNames VALUES (1, 1, "main:bidon");
INSERT INTO ProcessInputs VALUES (1, 1, "val", "arg1");
INSERT INTO ProcessInputs VALUES (1, 2, "path", "arg2");
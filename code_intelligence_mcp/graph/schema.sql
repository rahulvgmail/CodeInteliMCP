-- Code Intelligence Graph Schema for DuckDB

-- Enable JSON extension for metadata storage
INSTALL json;
LOAD json;

-- Repositories table: Track multiple codebases
CREATE SEQUENCE IF NOT EXISTS repositories_id_seq;
CREATE TABLE IF NOT EXISTS repositories (
    id INTEGER PRIMARY KEY DEFAULT nextval('repositories_id_seq'),
    name VARCHAR NOT NULL UNIQUE,
    path VARCHAR NOT NULL UNIQUE,
    description TEXT,
    language VARCHAR,                 -- Primary language of the repository
    metadata JSON,                    -- Git info, build system, etc.
    indexed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Symbols table: All code entities (functions, classes, variables, etc.)
CREATE SEQUENCE IF NOT EXISTS symbols_id_seq;
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY DEFAULT nextval('symbols_id_seq'),
    repository_id INTEGER NOT NULL,
    name VARCHAR NOT NULL,
    qualified_name VARCHAR NOT NULL,  -- Full path like module.class.method
    symbol_type VARCHAR NOT NULL,     -- 'function', 'class', 'method', 'variable', 'module', 'constant'
    file_path VARCHAR NOT NULL,       -- Relative to repository root
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    column_start INTEGER,
    column_end INTEGER,
    signature TEXT,                   -- Function/method signature
    docstring TEXT,                   -- Documentation string
    language VARCHAR NOT NULL,        -- Programming language
    visibility VARCHAR,               -- 'public', 'private', 'protected'
    is_async BOOLEAN DEFAULT FALSE,   -- For async functions
    is_generator BOOLEAN DEFAULT FALSE,
    complexity INTEGER,               -- Cyclomatic complexity
    ast_hash VARCHAR NOT NULL,        -- For change detection
    parent_symbol_id INTEGER,         -- For nested symbols (methods in classes)
    metadata JSON,                    -- Language-specific metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (repository_id) REFERENCES repositories(id),
    FOREIGN KEY (parent_symbol_id) REFERENCES symbols(id)
);

-- Relationships table: How symbols relate to each other
CREATE SEQUENCE IF NOT EXISTS relationships_id_seq;
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY DEFAULT nextval('relationships_id_seq'),
    from_symbol_id INTEGER NOT NULL,
    to_symbol_id INTEGER NOT NULL,
    relationship_type VARCHAR NOT NULL, -- 'calls', 'imports', 'inherits', 'implements', 'uses', 'instantiates', 'returns', 'throws'
    file_path VARCHAR NOT NULL,
    line_number INTEGER NOT NULL,
    column_number INTEGER,
    confidence FLOAT DEFAULT 1.0,     -- For fuzzy/uncertain relationships
    is_direct BOOLEAN DEFAULT TRUE,   -- Direct vs transitive relationship
    context TEXT,                     -- Code snippet showing the relationship
    metadata JSON,                    -- Additional relationship data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (from_symbol_id) REFERENCES symbols(id),
    FOREIGN KEY (to_symbol_id) REFERENCES symbols(id)
);

-- Files table: Track file-level information
CREATE SEQUENCE IF NOT EXISTS files_id_seq;
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY DEFAULT nextval('files_id_seq'),
    repository_id INTEGER NOT NULL,
    path VARCHAR NOT NULL,            -- Relative to repository root
    language VARCHAR NOT NULL,
    size_bytes INTEGER,
    line_count INTEGER,
    last_modified TIMESTAMP,
    content_hash VARCHAR NOT NULL,    -- For change detection
    imports JSON,                     -- List of imports/requires
    exports JSON,                     -- List of exports
    metrics JSON,                     -- File-level metrics
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (repository_id) REFERENCES repositories(id),
    UNIQUE(repository_id, path)       -- Path must be unique within repository
);

-- Modules table: Higher-level organization
CREATE SEQUENCE IF NOT EXISTS modules_id_seq;
CREATE TABLE IF NOT EXISTS modules (
    id INTEGER PRIMARY KEY DEFAULT nextval('modules_id_seq'),
    name VARCHAR NOT NULL,
    path VARCHAR NOT NULL UNIQUE,
    type VARCHAR,                     -- 'package', 'namespace', 'module'
    language VARCHAR NOT NULL,
    description TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Symbol metrics table: Performance and quality metrics
CREATE TABLE IF NOT EXISTS symbol_metrics (
    symbol_id INTEGER PRIMARY KEY,
    lines_of_code INTEGER,
    cyclomatic_complexity INTEGER,
    cognitive_complexity INTEGER,
    maintainability_index FLOAT,
    test_coverage FLOAT,
    change_frequency INTEGER,         -- How often it changes
    bug_frequency INTEGER,            -- Associated bug count
    last_measured TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_repositories_name ON repositories(name);
CREATE INDEX IF NOT EXISTS idx_repositories_path ON repositories(path);

CREATE INDEX IF NOT EXISTS idx_symbols_repo ON symbols(repository_id);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_qualified ON symbols(qualified_name);
CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols(symbol_type);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_symbol_id);
CREATE INDEX IF NOT EXISTS idx_symbols_hash ON symbols(ast_hash);
CREATE INDEX IF NOT EXISTS idx_symbols_repo_name ON symbols(repository_id, name);

CREATE INDEX IF NOT EXISTS idx_rel_from ON relationships(from_symbol_id);
CREATE INDEX IF NOT EXISTS idx_rel_to ON relationships(to_symbol_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_rel_confidence ON relationships(confidence);

CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repository_id);
CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_repo_path ON files(repository_id, path);

-- Create views for common queries
CREATE VIEW IF NOT EXISTS function_calls AS
SELECT 
    s1.name as caller_name,
    s1.file_path as caller_file,
    s1.line_start as caller_line,
    s2.name as callee_name,
    s2.file_path as callee_file,
    r.confidence
FROM relationships r
JOIN symbols s1 ON r.from_symbol_id = s1.id
JOIN symbols s2 ON r.to_symbol_id = s2.id
WHERE r.relationship_type = 'calls';

CREATE VIEW IF NOT EXISTS class_hierarchy AS
SELECT 
    s1.name as child_class,
    s1.file_path as child_file,
    s2.name as parent_class,
    s2.file_path as parent_file,
    r.relationship_type
FROM relationships r
JOIN symbols s1 ON r.from_symbol_id = s1.id
JOIN symbols s2 ON r.to_symbol_id = s2.id
WHERE r.relationship_type IN ('inherits', 'implements');

-- Materialized view for symbol dependencies (refreshed periodically)
CREATE TABLE IF NOT EXISTS symbol_dependencies AS
WITH RECURSIVE deps AS (
    -- Base case: direct dependencies
    SELECT 
        from_symbol_id as symbol_id,
        to_symbol_id as dependency_id,
        1 as depth,
        relationship_type,
        ARRAY[from_symbol_id, to_symbol_id] as path
    FROM relationships
    WHERE relationship_type IN ('calls', 'uses', 'imports')
    
    UNION ALL
    
    -- Recursive case: transitive dependencies
    SELECT 
        d.symbol_id,
        r.to_symbol_id as dependency_id,
        d.depth + 1 as depth,
        r.relationship_type,
        array_append(d.path, r.to_symbol_id) as path
    FROM deps d
    JOIN relationships r ON d.dependency_id = r.from_symbol_id
    WHERE d.depth < 5  -- Limit depth to prevent infinite recursion
    AND r.to_symbol_id != ALL(d.path)  -- Prevent cycles
)
SELECT DISTINCT 
    symbol_id,
    dependency_id,
    MIN(depth) as min_depth,
    COUNT(*) as path_count
FROM deps
GROUP BY symbol_id, dependency_id;

-- Function to get all callers of a symbol (recursive)
CREATE MACRO IF NOT EXISTS get_all_callers(target_symbol_id) AS TABLE
WITH RECURSIVE callers AS (
    SELECT 
        s.id,
        s.name,
        s.qualified_name,
        s.file_path,
        s.line_start,
        1 as depth
    FROM symbols s
    JOIN relationships r ON s.id = r.from_symbol_id
    WHERE r.to_symbol_id = target_symbol_id 
    AND r.relationship_type = 'calls'
    
    UNION ALL
    
    SELECT 
        s.id,
        s.name,
        s.qualified_name,
        s.file_path,
        s.line_start,
        c.depth + 1
    FROM symbols s
    JOIN relationships r ON s.id = r.from_symbol_id
    JOIN callers c ON r.to_symbol_id = c.id
    WHERE r.relationship_type = 'calls'
    AND c.depth < 10  -- Prevent runaway recursion
)
SELECT DISTINCT * FROM callers ORDER BY depth, name;

-- Function to find circular dependencies
CREATE MACRO IF NOT EXISTS find_circular_deps() AS TABLE
WITH RECURSIVE paths AS (
    SELECT 
        from_symbol_id as start_id,
        to_symbol_id as current_id,
        ARRAY[from_symbol_id, to_symbol_id] as path,
        1 as depth
    FROM relationships
    WHERE relationship_type IN ('calls', 'imports', 'uses')
    
    UNION ALL
    
    SELECT 
        p.start_id,
        r.to_symbol_id as current_id,
        array_append(p.path, r.to_symbol_id) as path,
        p.depth + 1
    FROM paths p
    JOIN relationships r ON p.current_id = r.from_symbol_id
    WHERE r.relationship_type IN ('calls', 'imports', 'uses')
    AND p.depth < 20
    AND r.to_symbol_id != ALL(p.path[2:])  -- Avoid revisiting except start
)
SELECT 
    path,
    array_to_string(path, ' -> ') as cycle
FROM paths
WHERE current_id = start_id
AND depth > 1;

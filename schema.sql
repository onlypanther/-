PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS history;
DROP TABLE IF EXISTS statistics;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);

CREATE TABLE history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    result TEXT NOT NULL,
    probability_good FLOAT,
    probability_bad FLOAT,
    probability_other FLOAT,
    image_path TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    total_scans INTEGER DEFAULT 0,
    good_results INTEGER DEFAULT 0,
    bad_results INTEGER DEFAULT 0,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- Insertar usuario de prueba
INSERT INTO users (username, password) VALUES ('admin', 'admin');
-- Crear estadísticas iniciales para el usuario admin
INSERT INTO statistics (user_id) VALUES (1);
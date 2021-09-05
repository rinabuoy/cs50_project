CREATE TABLE transactions (
	id INTEGER PRIMARY KEY,
	user_id INTEGER NOT NULL,
	symbol TEXT NOT NULL,
	shares INTEGER NOT NULL,
	price float NOT NULL,
	dt TEXT NOT NULL,
	tran_type TEXT NOT NULL,
    FOREIGN KEY (user_id)
       REFERENCES users (id)
);

CREATE INDEX symbol_index ON transactions (symbol);
CREATE INDEX dt_index ON transactions (dt);
CREATE INDEX type_index ON transactions (tran_type);

DROP INDEX symbol_index;
DROP INDEX dt_index;
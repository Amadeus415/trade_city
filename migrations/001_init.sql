-- Journal schema v1 — see AGENTIC_TRADING_PLAN §12

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  mode TEXT NOT NULL,
  started_at TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  git_sha TEXT NOT NULL,
  limits_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  as_of TEXT NOT NULL,
  session TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  model TEXT NOT NULL,
  temperature REAL NOT NULL,
  masking_mode TEXT NOT NULL,
  feature_packet_hash TEXT NOT NULL,
  feature_packet_json TEXT NOT NULL,
  raw_llm_response TEXT NOT NULL,
  latency_ms INTEGER,
  cost_usd REAL,
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS proposals (
  proposal_id TEXT PRIMARY KEY,
  decision_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  action TEXT NOT NULL,
  target_weight TEXT NOT NULL,
  confidence TEXT NOT NULL,
  thesis TEXT NOT NULL,
  invalidation TEXT NOT NULL,
  horizon_days INTEGER NOT NULL,
  source_features TEXT NOT NULL,
  FOREIGN KEY (decision_id) REFERENCES decisions(decision_id)
);

CREATE TABLE IF NOT EXISTS risk_decisions (
  proposal_id TEXT PRIMARY KEY,
  verdict TEXT NOT NULL,
  final_weight TEXT NOT NULL,
  reasons TEXT NOT NULL,
  limits_version TEXT NOT NULL,
  evaluated_at TEXT NOT NULL,
  FOREIGN KEY (proposal_id) REFERENCES proposals(proposal_id)
);

CREATE TABLE IF NOT EXISTS orders (
  client_order_id TEXT PRIMARY KEY,
  decision_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity TEXT NOT NULL,
  limit_price TEXT NOT NULL,
  status TEXT NOT NULL,
  broker_order_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  review_response TEXT,
  error TEXT
);

CREATE TABLE IF NOT EXISTS fills (
  fill_id TEXT PRIMARY KEY,
  client_order_id TEXT NOT NULL,
  quantity TEXT NOT NULL,
  price TEXT NOT NULL,
  fees TEXT NOT NULL,
  filled_at TEXT NOT NULL,
  FOREIGN KEY (client_order_id) REFERENCES orders(client_order_id)
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
  run_id TEXT NOT NULL,
  session TEXT NOT NULL,
  cash TEXT NOT NULL,
  equity TEXT NOT NULL,
  peak_equity TEXT NOT NULL,
  positions_json TEXT NOT NULL,
  PRIMARY KEY (run_id, session)
);

CREATE TABLE IF NOT EXISTS risk_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  state TEXT NOT NULL,
  since TEXT NOT NULL,
  trigger_metric TEXT,
  trigger_value TEXT,
  consecutive_loss_days INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS risk_state_transitions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  from_state TEXT NOT NULL,
  to_state TEXT NOT NULL,
  at TEXT NOT NULL,
  trigger_metric TEXT,
  trigger_value TEXT,
  operator_note TEXT
);

CREATE TABLE IF NOT EXISTS llm_cache (
  cache_key TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  response_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Singleton default risk state
INSERT OR IGNORE INTO risk_state (id, state, since, consecutive_loss_days)
VALUES (1, 'NORMAL', datetime('now'), 0);

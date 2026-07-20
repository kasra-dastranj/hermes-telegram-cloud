"use strict";

const path = require("path");

const dataDir = process.env.DATA_DIR || "/opt/data/9router";
const Database = require(
  path.join(dataDir, "runtime", "node_modules", "better-sqlite3")
);
const databasePath = path.join(dataDir, "db", "data.sqlite");
const db = new Database(databasePath);

const settings = JSON.stringify({ requireApiKey: false });
db.prepare(
  `INSERT INTO settings (id, data) VALUES (1, ?)
   ON CONFLICT(id) DO UPDATE SET data = excluded.data`
).run(settings);

const modelKey = "oc|deepseek-v4-flash-free|llm";
const modelValue = JSON.stringify({
  providerAlias: "oc",
  id: "deepseek-v4-flash-free",
  type: "llm",
  name: "deepseek-v4-flash-free",
});

db.prepare(
  `INSERT INTO kv (scope, key, value) VALUES ('customModels', ?, ?)
   ON CONFLICT(scope, key) DO UPDATE SET value = excluded.value`
).run(modelKey, modelValue);

db.close();
console.log("[startup] 9Router database configured without exposing its port.");

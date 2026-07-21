"use strict";

const fs = require("fs");
const path = require("path");
const { DatabaseSync } = require("node:sqlite");

const dataDir = process.env.DATA_DIR || "/opt/data/9router";
const databasePath = path.join(dataDir, "db", "data.sqlite");
if (!fs.existsSync(databasePath)) {
  throw new Error(`9Router did not create its database at ${databasePath}`);
}
const db = new DatabaseSync(databasePath);
db.exec("PRAGMA busy_timeout = 5000");

const freeModels = [
  "deepseek-v4-flash-free",
  "mimo-v2.5-free",
  "big-pickle",
  "nemotron-3-ultra-free",
  "north-mini-code-free",
];
const comboId = "hermes-free-fallback";
const comboName = "hermes-free";

try {
  db.exec("BEGIN IMMEDIATE");

  const currentRow = db
    .prepare("SELECT data FROM settings WHERE id = 1")
    .get();
  let currentSettings = {};
  if (currentRow?.data) {
    try {
      currentSettings = JSON.parse(currentRow.data);
    } catch {
      currentSettings = {};
    }
  }
  currentSettings.requireApiKey = false;

  db.prepare(
    `INSERT INTO settings (id, data) VALUES (1, ?)
     ON CONFLICT(id) DO UPDATE SET data = excluded.data`
  ).run(JSON.stringify(currentSettings));

  const upsertModel = db.prepare(
    `INSERT INTO kv (scope, key, value) VALUES ('customModels', ?, ?)
     ON CONFLICT(scope, key) DO UPDATE SET value = excluded.value`
  );
  for (const model of freeModels) {
    upsertModel.run(
      `oc|${model}|llm`,
      JSON.stringify({
        providerAlias: "oc",
        id: model,
        type: "llm",
        name: model,
      })
    );
  }

  const now = new Date().toISOString();
  db.prepare(
    `INSERT INTO combos (id, name, kind, models, createdAt, updatedAt)
     VALUES (?, ?, NULL, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       name = excluded.name,
       kind = excluded.kind,
       models = excluded.models,
       updatedAt = excluded.updatedAt`
  ).run(
    comboId,
    comboName,
    JSON.stringify(freeModels.map((model) => `oc/${model}`)),
    now,
    now
  );

  db.exec("COMMIT");
} catch (error) {
  try {
    db.exec("ROLLBACK");
  } catch {
    // The transaction may not have started; preserve the original error.
  }
  throw error;
} finally {
  db.close();
}
console.log(
  `[startup] 9Router configured with ${freeModels.length} OpenCode free models ` +
    `and fallback combo "${comboName}" without exposing its port.`
);

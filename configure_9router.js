"use strict";

const fs = require("fs");
const path = require("path");
const { DatabaseSync } = require("node:sqlite");

const dataDir = process.env.DATA_DIR || "/opt/data/9router";
const databasePath = path.join(dataDir, "db", "data.sqlite");

const openCodeModels = [
  "nemotron-3-ultra-free",
  "north-mini-code-free",
  "deepseek-v4-flash-free",
  "mimo-v2.5-free",
  "big-pickle",
];

const secretProviders = [
  {
    provider: "groq",
    envName: "GROQ_API_KEY",
    connectionId: "hermes-managed-groq",
    connectionName: "Hermes Groq Secret",
    models: [
      "openai/gpt-oss-120b",
      "llama-3.3-70b-versatile",
      "qwen/qwen3.6-27b",
    ],
  },
  {
    provider: "openrouter",
    envName: "OPENROUTER_API_KEY",
    connectionId: "hermes-managed-openrouter",
    connectionName: "Hermes OpenRouter Secret",
    models: ["openrouter/free"],
  },
];

const comboId = "hermes-free-fallback";
const comboName = "hermes-free";

function configureDatabase(enabledProviders) {
  if (!fs.existsSync(databasePath)) {
    throw new Error(`9Router did not create its database at ${databasePath}`);
  }

  const db = new DatabaseSync(databasePath);
  db.exec("PRAGMA busy_timeout = 5000");

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

    const deleteManagedConnection = db.prepare(
      "DELETE FROM providerConnections WHERE id = ?"
    );
    for (const spec of secretProviders) {
      deleteManagedConnection.run(spec.connectionId);
    }

    const upsertModel = db.prepare(
      `INSERT INTO kv (scope, key, value) VALUES ('customModels', ?, ?)
       ON CONFLICT(scope, key) DO UPDATE SET value = excluded.value`
    );

    const upsertConnection = db.prepare(
      `INSERT INTO providerConnections
         (id, provider, authType, name, email, priority, isActive, data, createdAt, updatedAt)
       VALUES (?, ?, 'apikey', ?, NULL, 1, 1, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         provider = excluded.provider,
         authType = excluded.authType,
         name = excluded.name,
         priority = excluded.priority,
         isActive = excluded.isActive,
         data = excluded.data,
         updatedAt = excluded.updatedAt`
    );

    const comboModels = [];
    for (const model of openCodeModels) {
      upsertModel.run(
        `oc|${model}|llm`,
        JSON.stringify({
          providerAlias: "oc",
          id: model,
          type: "llm",
          name: model,
        })
      );
      comboModels.push(`oc/${model}`);
    }

    for (const spec of enabledProviders) {
      const now = new Date().toISOString();
      db.prepare(
        `DELETE FROM providerConnections
         WHERE provider = ? AND name = ? AND id <> ?`
      ).run(spec.provider, spec.connectionName, spec.connectionId);
      upsertConnection.run(
        spec.connectionId,
        spec.provider,
        spec.connectionName,
        JSON.stringify({
          apiKey: spec.apiKey,
          testStatus: "unknown",
        }),
        now,
        now
      );

      for (const model of spec.models) {
        upsertModel.run(
          `${spec.provider}|${model}|llm`,
          JSON.stringify({
            providerAlias: spec.provider,
            id: model,
            type: "llm",
            name: model,
          })
        );
        comboModels.push(`${spec.provider}/${model}`);
      }
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
    ).run(comboId, comboName, JSON.stringify(comboModels), now, now);

    db.exec("COMMIT");
    return comboModels;
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
}

function main() {
  const enabledProviders = [];

  for (const spec of secretProviders) {
    const apiKey = String(process.env[spec.envName] || "").trim();
    if (!apiKey) {
      console.warn(
        `[startup] ${spec.envName} is not set; omitting ${spec.provider} from the text fallback combo.`
      );
      continue;
    }

    enabledProviders.push({ ...spec, apiKey });
    console.log(
      `[startup] Loaded ${spec.provider} connection from deployment secrets.`
    );
  }

  const comboModels = configureDatabase(enabledProviders);
  console.log(
    `[startup] 9Router configured fallback combo "${comboName}" with ` +
      `${comboModels.length} models across ${enabledProviders.length + 1} providers ` +
      `without exposing its port or secrets.`
  );
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error(`[startup] Failed to configure 9Router: ${error.message}`);
    process.exitCode = 1;
  }
}

module.exports = {
  comboName,
  configureDatabase,
  openCodeModels,
  secretProviders,
};

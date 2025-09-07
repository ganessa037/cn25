import express from "express";
import { prisma } from "../db/prisma.js";
import { requireUser } from "../middleware/requireUser.js";

const r = express.Router();
r.use(requireUser);

/* --------------------------- helpers --------------------------- */

// Keep only whitelisted keys
function pickVehicle(body = {}) {
  const out = {};
  const allow = new Set([
    "brand",
    "model",
    "year",
    "plate",
    "color",
    "fuelType",
    "chassisNumber",
    "engineNumber",
    "roadTaxExpiry",
    "insuranceExpiry",
    "lastServiceDate",
    "nextServiceDate",
    "currentMileage",
  ]);
  for (const k of Object.keys(body)) {
    if (allow.has(k)) out[k] = body[k];
  }
  return out;
}

// Accept "" -> null and normalize to "YYYY-MM-DDT00:00:00.000"
function toDbTimestamp(v) {
  if (v === undefined || v === null || v === "") return null;
  // support already-sent ISO
  if (typeof v === "string" && /T\d{2}:\d{2}/.test(v)) {
    const d = new Date(v);
    return isNaN(+d) ? null : d;
  }
  // support YYYY-MM-DD
  if (typeof v === "string" && /^\d{4}-\d{2}-\d{2}$/.test(v)) {
    return new Date(`${v}T00:00:00.000`);
  }
  const d = new Date(v);
  return isNaN(+d) ? null : d;
}

function coercePatch(patch) {
  const out = pickVehicle(patch);

  // Numeric coercions
  if ("year" in out) {
    out.year = out.year === "" ? null : Number(out.year);
    if (out.year !== null && Number.isNaN(out.year)) out.year = null;
  }
  if ("currentMileage" in out) {
    out.currentMileage =
      out.currentMileage === "" || out.currentMileage === undefined
        ? undefined
        : Number(out.currentMileage);
    if (Number.isNaN(out.currentMileage)) out.currentMileage = undefined;
  }

  // Date fields
  ["roadTaxExpiry", "insuranceExpiry", "lastServiceDate", "nextServiceDate"].forEach(
    (k) => {
      if (k in out) out[k] = toDbTimestamp(out[k]);
    }
  );

  // Trim simple strings
  ["brand", "model", "plate", "color", "fuelType", "chassisNumber", "engineNumber"].forEach(
    (k) => {
      if (k in out && typeof out[k] === "string") {
        out[k] = out[k].trim();
        if (out[k] === "") out[k] = null;
      }
    }
  );

  return out;
}

/* ----------------------------- routes ----------------------------- */

/** GET /api/vehicles (current user only) */
r.get("/", async (req, res, next) => {
  try {
    const rows = await prisma.vehicle.findMany({
      where: { userId: req.user.id },
      orderBy: [{ updatedAt: "desc" }, { createdAt: "desc" }],
    });
    res.json(rows);
  } catch (err) {
    next(err);
  }
});

/** GET /api/vehicles/:id (ownership enforced) */
r.get("/:id", async (req, res, next) => {
  try {
    const row = await prisma.vehicle.findFirst({
      where: { id: req.params.id, userId: req.user.id },
    });
    if (!row) return res.status(404).json({ error: "Not found" });
    res.json(row);
  } catch (err) {
    next(err);
  }
});

/** POST /api/vehicles (whitelist + required brand) */
r.post("/", async (req, res, next) => {
  try {
    const data = coercePatch(req.body);
    if (!data.brand) {
      return res.status(400).json({ error: "Field 'brand' is required." });
    }
    if (data.currentMileage === undefined) data.currentMileage = 0;

    const created = await prisma.vehicle.create({
      data: { ...data, userId: req.user.id },
    });
    res.status(201).json(created);
  } catch (err) {
    next(err);
  }
});

/** PUT /api/vehicles/:id (ownership + whitelist) */
r.put("/:id", async (req, res, next) => {
  try {
    // Ensure the vehicle belongs to this user
    const found = await prisma.vehicle.findFirst({
      where: { id: req.params.id, userId: req.user.id },
      select: { id: true },
    });
    if (!found) return res.status(404).json({ error: "Not found" });

    const data = coercePatch(req.body);

    const updated = await prisma.vehicle.update({
      where: { id: found.id },
      data,
    });
    res.json(updated);
  } catch (err) {
    next(err);
  }
});

/** DELETE /api/vehicles/:id (ownership) */
r.delete("/:id", async (req, res, next) => {
  try {
    const found = await prisma.vehicle.findFirst({
      where: { id: req.params.id, userId: req.user.id },
      select: { id: true },
    });
    if (!found) return res.status(404).json({ error: "Not found" });

    await prisma.vehicle.delete({ where: { id: found.id } });
    res.json({ success: true });
  } catch (err) {
    next(err);
  }
});

export default r;
import express from "express";
import fs from "fs/promises";
import { prisma } from "../db/prisma.js";
import { requireUser } from "../middleware/requireUser.js";
import { upload } from "../middleware/upload.js";
import { extractTextFromBuffer } from "../services/mlService.js";

const r = express.Router();
r.use(requireUser);

/** GET /api/documents (current user only) */
r.get("/", async (req, res) => {
  const rows = await prisma.document.findMany({
    where: { userId: req.user.id },
    orderBy: { uploadedAt: "desc" },
  });
  res.json(rows);
});

/**
 * POST /api/documents
 * multipart/form-data fields:
 *   - file (required)
 *   - type (optional enum)
 *   - name (required)
 *   - vehicleId (optional, must belong to user)
 *   - expiryDate (optional, ISO or YYYY-MM-DDTHH:mm:ss.SSS)
 */
r.post("/", upload.single("file"), async (req, res) => {
  try {
    const { type, name, vehicleId, expiryDate } = req.body;
    if (!name || !req.file) {
      return res.status(400).json({ error: "Missing required fields: name and file." });
    }

    // Validate optional vehicle ownership
    let vehicleIdSafe = null;
    if (vehicleId) {
      const owned = await prisma.vehicle.findFirst({
        where: { id: String(vehicleId), userId: req.user.id },
        select: { id: true },
      });
      if (owned) vehicleIdSafe = owned.id;
    }

    // Read & encode file
    const fileBuf = await fs.readFile(req.file.path);
    const size = req.file.size ?? fileBuf.length;
    const contentBase64 = `data:${req.file.mimetype};base64,${fileBuf.toString("base64")}`;

    // Create the row first (fast response path)
    const created = await prisma.document.create({
      data: {
        userId: req.user.id,
        vehicleId: vehicleIdSafe,
        type: type || null,
        name: name.trim(),
        expiryDate: expiryDate ? new Date(expiryDate) : null,
        uploadedAt: new Date(),
        size,
        contentBase64,
        extractedText: null, // to be filled asynchronously
      },
    });

    // Fire-and-forget OCR so the client gets a quick 201
    (async () => {
      try {
        const text = await extractTextFromBuffer(fileBuf, req.file.mimetype);
        if (text && text.length > 0) {
          await prisma.document.update({
            where: { id: created.id },
            data: { extractedText: text },
          });
        }
      } catch (err) {
        console.error("OCR failed:", err);
      } finally {
        // Clean up temp file
        try {
          await fs.unlink(req.file.path);
        } catch {}
      }
    })();

    return res.status(201).json(created);
  } catch (err) {
    console.error("document upload error:", err);
    return res.status(500).json({ error: "Upload failed" });
  }
});

/** PUT /api/documents/:id (metadata only) */
r.put("/:id", async (req, res) => {
  const { id } = req.params;
  const b = req.body || {};
  const d = await prisma.document.update({
    where: { id },
    data: {
      type: b.type ?? undefined,
      name: b.name?.trim?.() ?? undefined,
      vehicleId: b.vehicleId ?? undefined,
      expiryDate: b.expiryDate ? new Date(b.expiryDate) : null,
      extractedText: b.extractedText ?? undefined,
    },
  });
  res.json(d);
});

/** DELETE /api/documents/:id */
r.delete("/:id", async (req, res) => {
  await prisma.document.delete({ where: { id: req.params.id } });
  res.json({ success: true });
});

export default r;
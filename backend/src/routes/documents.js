import express from "express";
import fs from "fs/promises";
import Tesseract from "tesseract.js";

import { prisma } from "../db/prisma.js";
import { requireUser } from "../middleware/requireUser.js";
import { upload } from "../middleware/upload.js"; // your existing multer (disk)

const router = express.Router();
router.use(requireUser);

/** ---- helpers ---- */
function parseMaybeDate(input) {
  if (!input) return null;
  const iso = new Date(input);
  if (!Number.isNaN(+iso)) return iso;
  const m = String(input).match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
  if (m) {
    const [, dd, mm, yyyy] = m;
    const d = new Date(Number(yyyy), Number(mm) - 1, Number(dd));
    if (!Number.isNaN(+d)) return d;
  }
  return null;
}

/**
 * Demo OCR/Text extraction (IMAGE-ONLY to avoid pdf-parse crash):
 *  - Images → tesseract.js (first run may take a few seconds)
 *  - PDFs → returns null (no text extraction for demo)
 * Returns a trimmed string or null. Never throws.
 */
async function extractTextFromUpload(file) {
  try {
    let buffer = file?.buffer;
    if (!buffer && file?.path) buffer = await fs.readFile(file.path);
    if (!buffer) return null;

    const mime = file.mimetype || "";

    // Image OCR
    if (mime.startsWith("image/")) {
      const { data } = await Tesseract.recognize(buffer, "eng");
      const text = (data?.text || "").trim();
      return text || null;
    }

    // PDFs and other types: skip OCR for demo (avoid pdf-parse)
    return null;
  } catch {
    return null;
  }
}

/** ---- routes ---- */

// List user documents
router.get("/", async (req, res, next) => {
  try {
    const docs = await prisma.document.findMany({
      where: { userId: req.user.id },
      orderBy: { uploadedAt: "desc" },
    });
    res.json(docs);
  } catch (err) {
    next(err);
  }
});

// Create + local OCR (image-only demo)
router.post("/", upload.single("file"), async (req, res, next) => {
  try {
    const userId = req.user.id;
    const { type, name, vehicleId, expiryDate } = req.body;
    const f = req.file;
    if (!f) return res.status(400).json({ error: "file is required" });

    // best-effort OCR (image only)
    const ocrText = await extractTextFromUpload(f);

    const created = await prisma.document.create({
      data: {
        userId,
        type,               // Prisma enum e.g. 'LICENSE' | 'INSURANCE' | ...
        name,
        vehicleId: vehicleId || null,
        expiryDate: parseMaybeDate(expiryDate),
        size: f.size,

        // Uncomment if your schema has these:
        // mimeType: f.mimetype,
        // originalName: f.originalname,
        // storagePath: f.path,

        extractedText: ocrText,   // <-- ensure this matches your Prisma field
      },
    });

    res.status(201).json(created);
  } catch (err) {
    next(err);
  }
});

// Update metadata
router.put("/:id", async (req, res, next) => {
  try {
    const data = { ...req.body };
    if ("expiryDate" in data) data.expiryDate = parseMaybeDate(data.expiryDate);

    const updated = await prisma.document.update({
      where: { id: req.params.id, userId: req.user.id },
      data,
    });
    res.json(updated);
  } catch (err) {
    next(err);
  }
});

// Delete
router.delete("/:id", async (req, res, next) => {
  try {
    await prisma.document.delete({
      where: { id: req.params.id, userId: req.user.id },
    });
    res.json({ success: true });
  } catch (err) {
    next(err);
  }
});

export default router;
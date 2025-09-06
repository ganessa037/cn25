import * as models from "../models/index.js";
const { Document } = models;

export async function createDocument(req, res, next) {
  try {
    const { vehicleId } = req.body;
    const { filename, mimetype, size, path } = req.file;
    const doc = await Document.create({
      vehicleId, filename, mimetype, size, path, status: "UPLOADED",
      ocrText: null, ocrConfidence: null
    });
    res.status(201).json({
      documentId: doc.id, status: doc.status, mime: mimetype, size,
      ocr: { text: null, confidence: null }
    });
  } catch (e) { next(e); }
}
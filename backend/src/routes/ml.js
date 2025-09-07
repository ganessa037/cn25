import { Router } from "express";
import { requireUser } from "../middleware/requireUser.js";
import { upload } from "../middleware/upload.js"; // multer instance
import { mlHealth, mlPredict, mlExtractText, mlTrain, mlJobStatus } from "../services/mlService.js";

const router = Router();
router.use(requireUser); // protect all ML routes

router.get("/health", async (req, res, next) => {
  try { res.json(await mlHealth()); } catch (e) { next(e); }
});

router.post("/predict", async (req, res, next) => {
  try {
    const { text } = req.body;
    res.json(await mlPredict({ text }));
  } catch (e) { next(e); }
});

router.post("/extract-text", upload.single("file"), async (req, res, next) => {
  try {
    const f = req.file;
    if (!f) return res.status(400).json({ error: "file is required" });
    const out = await mlExtractText(f.buffer, f.originalname, f.mimetype);
    res.json(out);
  } catch (e) { next(e); }
});

router.post("/train", async (req, res, next) => {
  try { res.json(await mlTrain(req.body)); } catch (e) { next(e); }
});

router.get("/jobs/:id", async (req, res, next) => {
  try { res.json(await mlJobStatus(req.params.id)); } catch (e) { next(e); }
});

export default router;
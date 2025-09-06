import express from "express";
import passport from "passport";
import { initGooglePassport, startGoogle, googleCallback } from "../controllers/authController.js";

const router = express.Router();
initGooglePassport();
router.use(passport.initialize());

router.get("/google/start", startGoogle);
router.get("/google/callback", ...googleCallback);

export default router;
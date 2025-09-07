// Google OAuth via Passport; creates/loads a Sequelize user,
// mints a JWT, and redirects back to frontend with ?token=...&next=...
import passport from "passport";
import { Strategy as GoogleStrategy } from "passport-google-oauth20";
import jwt from "jsonwebtoken";
import { randomBytes } from "crypto";
import config from "../config/config.js";
import logger from "../utils/logger.js";
import * as models from "../models/index.js";

/** Build the frontend redirect URL with token and optional next */
function buildFrontendRedirect(token, nextOverride) {
  const base = `${config.frontendUrl.replace(/\/$/, "")}${config.postLoginPath}`;
  const usp = new URLSearchParams({ token });
  if (nextOverride) usp.set("next", nextOverride);
  return `${base}?${usp.toString()}`;
}

/** Normalize names from Google profile */
function extractNames(profile) {
  const given =
    profile?.name?.givenName ||
    (profile?.displayName ? String(profile.displayName).split(" ")[0] : "") ||
    "Google";
  const family =
    profile?.name?.familyName ||
    (profile?.displayName
      ? String(profile.displayName).split(" ").slice(1).join(" ")
      : "") ||
    "User";
  return { given, family };
}

/** Initialize the Google strategy (stateless) */
export function initGooglePassport() {
  const { clientID, clientSecret, callbackURL } = config.google;

  if (!clientID || !clientSecret || !callbackURL) {
    logger.error("[OAuth] Missing Google env (GOOGLE_CLIENT_ID/SECRET/CALLBACK_URL)");
    throw new Error("Google OAuth is not configured");
  }

  passport.use(
    new GoogleStrategy(
      {
        clientID,
        clientSecret,
        callbackURL, // must exactly match the Google Console value
        passReqToCallback: true,
      },
      // Verify callback: find or create a Sequelize user
      async (_req, _accessToken, _refreshToken, profile, done) => {
        try {
          const email =
            profile?.emails?.find((e) => e.verified)?.value ||
            profile?.emails?.[0]?.value;
        
          if (!email) return done(new Error("No email returned by Google"));
        
          const displayName = profile?.displayName || email.split("@")[0];
          const avatar = profile?.photos?.[0]?.value || null;
        
          // Find by email
          let user = await models.User.findOne({ where: { email } });
        
          if (!user) {
            user = await models.User.create({
              email,
              name: displayName,
              avatarUrl: avatar,
            });
          } else {
            // best-effort profile refresh
            const patch = {};
            if (!user.name && displayName) patch.name = displayName;
            if (!user.avatarUrl && avatar) patch.avatarUrl = avatar;
            if (Object.keys(patch).length) await user.update(patch);
          }
        
          return done(null, {
            id: user.id,
            email: user.email,
            name: user.name || email,
          });
        } catch (err) {
          logger.error("[OAuth] Verify error:", err);
          return done(err);
        }
      }
    )
  );
}

/** Start OAuth; carries optional ?next=... in state */
export function startGoogle(req, res, next) {
  const nextParam = typeof req.query.next === "string" ? req.query.next : "";
  const state = nextParam ? `n:${encodeURIComponent(nextParam)}` : "";
  const authenticator = passport.authenticate("google", {
    scope: ["profile", "email"],
    prompt: "select_account",
    session: false,
    state,
  });
  authenticator(req, res, next);
}

/** Callback: authenticate, mint JWT, and redirect back */
export const googleCallback = [
  passport.authenticate("google", {
    session: false,
    failureRedirect: `${config.frontendUrl.replace(/\/$/, "")}/signin?error=oauth_failed`,
  }),
  async (req, res) => {
    try {
      let nextOverride = "";
      if (typeof req.query.state === "string" && req.query.state.startsWith("n:")) {
        nextOverride = decodeURIComponent(req.query.state.slice(2));
      }

      const user = req.user;
      if (!user?.id) {
        return res.redirect(
          `${config.frontendUrl.replace(/\/$/, "")}/signin?error=missing_user`
        );
      }

      const token = jwt.sign(
        { userId: user.id, email: user.email },
        config.jwt.secret,
        { expiresIn: "12h" }
      );

      const redirectUrl = buildFrontendRedirect(token, nextOverride);
      return res.redirect(302, redirectUrl);
    } catch (err) {
      logger.error("[OAuth] Callback error:", err);
      return res.redirect(
        `${config.frontendUrl.replace(/\/$/, "")}/signin?error=oauth_exception`
      );
    }
  },
];
import passport from "passport";
import { Strategy as GoogleStrategy } from "passport-google-oauth20";
import jwt from "jsonwebtoken";
import { prisma } from "../db/prisma.js";

const FRONTEND_URL = process.env.FRONTEND_URL;
const CALLBACK_URL =
  process.env.GOOGLE_CALLBACK_URL ||
  "http://127.0.0.1:3000/api/auth/google/callback";

// 仅初始化一次 Google 策略
export function initGooglePassport() {
  const { GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET } = process.env;
  if (!GOOGLE_CLIENT_ID || !GOOGLE_CLIENT_SECRET) {
    console.error("❌ Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET");
    process.exit(1);
  }

  passport.use(
    new GoogleStrategy(
      {
        clientID: GOOGLE_CLIENT_ID,
        clientSecret: GOOGLE_CLIENT_SECRET,
        callbackURL: CALLBACK_URL,
      },
      async (_accessToken, _refreshToken, profile, done) => {
        try {
          const email = profile.emails?.[0]?.value;
          const name = profile.displayName;
          const avatarUrl = profile.photos?.[0]?.value;
          if (!email) return done(new Error("No email in Google profile"));

          const user = await prisma.user.upsert({
            where: { email },
            update: { name, avatarUrl },
            create: { email, name, avatarUrl },
          });

          return done(null, {
            id: user.id,
            email: user.email,
            name: user.name,
            avatarUrl,
          });
        } catch (e) {
          return done(e);
        }
      }
    )
  );
}

// 起跳到 Google
export const startGoogle = passport.authenticate("google", {
  scope: ["profile", "email"],
  prompt: "select_account",
  session: false,
});

// 回调：签发 JWT 并重定向前端
export const googleCallback = [
  passport.authenticate("google", {
    session: false,
    failureRedirect: `${FRONTEND_URL}/signin`,
  }),
  async (req, res) => {
    const payload = {
      userId: req.user.id,
      email: req.user.email,
      name: req.user.name,
      avatarUrl: req.user.avatarUrl,
    };
    const token = jwt.sign(payload, process.env.JWT_SECRET || "dev-secret", {
      expiresIn: "7d",
    });
    // 关键：统一从这里重定向携带 token
    res.redirect(`${FRONTEND_URL}/dashboard?token=${encodeURIComponent(token)}`);
  },
];
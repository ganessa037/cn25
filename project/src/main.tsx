import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { GoogleOAuthProvider } from "@react-oauth/google";
import App from "./App";
import ErrorBoundary from "./shared/ErrorBoundary";
import "./index.css";

// Read the client id from Vite env
const GOOGLE_CLIENT_ID =
  (import.meta as any).env?.VITE_GOOGLE_CLIENT_ID as string | undefined;

/**
 * Render children with GoogleOAuthProvider only when client id exists.
 * If it's missing, we still render the app so the site isn't blank,
 * and log a clear warning.
 */
function WithGoogle({ children }: { children: React.ReactNode }) {
  if (!GOOGLE_CLIENT_ID) {
    console.warn(
      "[Google OAuth] Missing VITE_GOOGLE_CLIENT_ID in your environment (.env). " +
      "Google login will be disabled until you set it."
    );
    return <>{children}</>;
  }
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      {children}
    </GoogleOAuthProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <WithGoogle>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </WithGoogle>
    </ErrorBoundary>
  </React.StrictMode>
);
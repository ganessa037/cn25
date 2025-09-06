import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import { GoogleOAuthProvider } from "@react-oauth/google";

const cid = import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined;

const rootEl = document.getElementById("root");
if (!rootEl) {
  const fallback = document.createElement("div");
  fallback.style.cssText = "padding:16px;color:#fff;background:#111;min-height:100vh";
  fallback.innerHTML =
    "<h1 style='font-size:20px;margin-bottom:8px'>Missing #root</h1><p>index.html need &lt;div id='root'&gt;&lt;/div&gt;</p>";
  document.body.appendChild(fallback);
} else {
  ReactDOM.createRoot(rootEl).render(
    <React.StrictMode>
      {cid ? (
        <GoogleOAuthProvider clientId={cid}>
          <App />
        </GoogleOAuthProvider>
      ) : (
        <App />
      )}
    </React.StrictMode>
  );
}
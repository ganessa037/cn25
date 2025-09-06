import React from "react";
import clsx from "clsx";

export const GlassCard: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, ...props }) => (
  <div className={clsx("glass-card", className)} {...props} />
);

export const GlassPanel: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className, ...props }) => (
  <div className={clsx("glass-panel", className)} {...props} />
);

export const GlassButton: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement>> = ({ className, ...props }) => (
  <button className={clsx("glass-btn", className)} {...props} />
);

export const GlassInput = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => <input ref={ref} className={clsx("glass-input", className)} {...props} />
);
GlassInput.displayName = "GlassInput";
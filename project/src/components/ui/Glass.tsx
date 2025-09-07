import React from "react";

type BoxProps = React.HTMLAttributes<HTMLDivElement>;
type BtnProps = React.ButtonHTMLAttributes<HTMLButtonElement>;

// Page wrapper (optional utility for large sections)
export function GlassPage({ className = "", ...rest }: BoxProps) {
  return (
    <div className={`min-h-[calc(100vh-4rem)] ${className}`} {...rest} />
  );
}

// Section title helper
export function SectionTitle({ className = "", ...rest }: BoxProps) {
  return (
    <h2
      className={`text-xl font-semibold tracking-wide ${className}`}
      {...rest}
    />
  );
}

// Simple responsive grid helper
export function GlassGrid({ className = "", ...rest }: BoxProps) {
  return (
    <div
      className={`grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 ${className}`}
      {...rest}
    />
  );
}

// Single, canonical GlassCard export
export function GlassCard({ className = "", ...rest }: BoxProps) {
  return <div className={`glass-card ${className}`} {...rest} />;
}

// Single, canonical GlassPanel export
export function GlassPanel({ className = "", ...rest }: BoxProps) {
  return <div className={`glass-panel ${className}`} {...rest} />;
}

// Single, canonical GlassButton export
export function GlassButton({ className = "", ...rest }: BtnProps) {
  return (
    <button className={`glass-btn inline-flex items-center justify-center ${className}`} {...rest} />
  );
}
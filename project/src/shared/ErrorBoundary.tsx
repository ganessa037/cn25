import React from "react";

type Props = { children: React.ReactNode };
type State = { err?: Error };

export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = {};
  static getDerivedStateFromError(err: Error) { return { err }; }
  componentDidCatch(err: Error, info: React.ErrorInfo) {
    console.error("[App crash]", err, info);
  }
  render() {
    if (this.state.err) {
      return (
        <div className="min-h-screen grid place-items-center bg-black text-white">
          <div className="max-w-xl w-full p-6 rounded-2xl bg-white/10 border border-white/20 backdrop-blur-xl">
            <div className="text-lg font-semibold mb-2">Something went wrong.</div>
            <pre className="text-sm whitespace-pre-wrap opacity-80">
{this.state.err.message}
            </pre>
            <div className="text-xs opacity-60 mt-3">
              Check DevTools → Console for the stack trace.
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
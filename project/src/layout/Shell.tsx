import { Outlet } from "react-router-dom";
import Header from "./Header";
import Footer from "./Footer";

export default function Shell() {
  return (
    <div className="min-h-screen flex flex-col bg-neutral-950 text-neutral-100">
      <Header />
      <main className="flex-1">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { installZoomGuard } from "./lib/zoom";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/ui.css";
import "./styles/app.css";

installZoomGuard();

const root = document.getElementById("root");
if (!root) throw new Error("Root element is missing");

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// Entry point: routes for every page of the dashboard.
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import RoomLayout from "./components/RoomLayout.jsx";
import Deploy from "./pages/Deploy.jsx";
import Home from "./pages/Home.jsx";
import Join from "./pages/Join.jsx";
import Network from "./pages/Network.jsx";
import Nodes from "./pages/Nodes.jsx";
import Reports from "./pages/Reports.jsx";
import Training from "./pages/Training.jsx";
import "./styles.css";
import { applyTheme, savedTheme } from "./components/ThemeToggle.jsx";

applyTheme(savedTheme());

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/join/:code" element={<Join />} />
        <Route path="/room/:code" element={<RoomLayout />}>
          <Route index element={<Navigate to="nodes" replace />} />
          <Route path="nodes" element={<Nodes />} />
          <Route path="training" element={<Training />} />
          <Route path="network" element={<Network />} />
          <Route path="deploy" element={<Deploy />} />
          <Route path="reports" element={<Reports />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);

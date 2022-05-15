import { Routes, Route } from "react-router-dom";

import Search from "./Pages/Search/Search";
import Results from "./Pages/Results/Results";

function App() {
  return (
    <div className="app">
      <Routes>
        <Route path="/" element={<Search />} />
        <Route path='/:searchParam' element={<Results />} />
      </Routes>
    </div>
  );
}

export default App;

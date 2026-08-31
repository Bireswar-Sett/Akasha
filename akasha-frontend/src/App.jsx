import "./index.css";
import Navbar from "./components/Navbar";
import Analysis from "./pages/Analysis";

function App() {
  return (
    <div className="app">
      <Navbar />

      <main>
        <Analysis />
      </main>
    </div>
  );
}

export default App;
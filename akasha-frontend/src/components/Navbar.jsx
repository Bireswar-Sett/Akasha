function Navbar() {
  return (
    <nav className="navbar">
      <div className="brand">
        <div className="brand-icon">◉</div>

        <div>
          <div className="brand-name">AKASHA</div>
          <div className="brand-subtitle">Earth Observation AI</div>
        </div>
      </div>

      <div className="nav-links">
        <button className="nav-link active">Analyze</button>
        <button className="nav-link">History</button>
      </div>
    </nav>
  );
}

export default Navbar;
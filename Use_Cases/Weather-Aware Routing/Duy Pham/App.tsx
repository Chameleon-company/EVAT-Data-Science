import React from 'react';
import MapComponent from './MapComponent';
import './App.css';

const GOOGLE_MAPS_API_KEY = process.env.REACT_APP_GOOGLE_MAPS_API_KEY || '';

const App: React.FC = () => {
  return (
    <div className="App">
      <header className="App-header" style={{ padding: '20px', backgroundColor: '#f5f5f5' }}>
        <h1 style={{ margin: '0 0 10px 0', color: '#333' }}>
          EV Traffic & Weather Prediction
        </h1>
        <p style={{ margin: 0, color: '#666', fontSize: '14px' }}>
          Click on the map to select a location, then click "Calculate Energy" to get predictions
        </p>
      </header>
      
      <main style={{ width: '100%', height: 'calc(100vh - 120px)' }}>
        <MapComponent googleMapsApiKey={GOOGLE_MAPS_API_KEY} />
      </main>
    </div>
  );
};

export default App;

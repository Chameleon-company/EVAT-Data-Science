import React from 'react';
import MapComponent from './MapComponent';

const App: React.FC = () => {
  const GOOGLE_MAPS_API_KEY = process.env.REACT_APP_GOOGLE_MAPS_API_KEY || 'YOUR_API_KEY_HERE';

  return (
    <div style={{ padding: '20px' }}>
      <h1 style={{ marginBottom: '20px' }}>EV Traffic & Weather Prediction</h1>
      <MapComponent googleMapsApiKey={GOOGLE_MAPS_API_KEY} />
    </div>
  );
};

export default App;

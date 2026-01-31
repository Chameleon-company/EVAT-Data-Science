import React, { useState, useCallback } from 'react';
import { GoogleMap, Marker, useJsApiLoader } from '@react-google-maps/api';
import { predictEVTraffic, PredictionResponse, ApiError } from './apiClient';

const mapContainerStyle = {
  width: '100%',
  height: '600px',
};

const defaultCenter = {
  lat: -37.8136,
  lng: 144.9631,
};

const mapOptions = {
  disableDefaultUI: false,
  zoomControl: true,
  streetViewControl: false,
  mapTypeControl: true,
  fullscreenControl: true,
};

interface MapComponentProps {
  googleMapsApiKey?: string;
}

const MapComponent: React.FC<MapComponentProps> = ({ googleMapsApiKey = '' }) => {
  const [startLocation, setStartLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mapCenter, setMapCenter] = useState(defaultCenter);

  const { isLoaded, loadError } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: googleMapsApiKey || '',
    preventGoogleFontsLoading: false,
  });

  const handleMapClick = useCallback((event: google.maps.MapMouseEvent) => {
    if (event.latLng) {
      const lat = event.latLng.lat();
      const lng = event.latLng.lng();
      setStartLocation({ lat, lng });
      setPredictionResult(null);
      setError(null);
    }
  }, []);

  const handlePredictEnergy = useCallback(async () => {
    if (!startLocation) {
      setError('Please click on the map to select a start location');
      return;
    }

    setLoading(true);
    setError(null);
    setPredictionResult(null);

    try {
      const result = await predictEVTraffic({
        year: 2026,
        start_lat: startLocation.lat,
        start_lon: startLocation.lng,
      });
      setPredictionResult(result);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.error || 'Failed to get prediction');
      console.error('Prediction error:', apiError);
    } finally {
      setLoading(false);
    }
  }, [startLocation]);

  const handleReset = useCallback(() => {
    setStartLocation(null);
    setPredictionResult(null);
    setError(null);
    setMapCenter(defaultCenter);
  }, []);

  if (loadError) {
    const msg = loadError.message || '';
    const isKeyError = msg.includes('key') || msg.includes('API') || msg.includes('billing') || msg.includes('quota');
    if (!isKeyError && msg) {
      console.warn('Google Maps load error:', loadError);
    }
  }

  if (!isLoaded) {
    return (
      <div className="loading-container">
        <p>Loading Google Maps...</p>
        {!googleMapsApiKey && (
          <p style={{ fontSize: '12px', marginTop: '10px', color: '#999' }}>
            Development mode: Map will show a watermark
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="map-container" style={{ position: 'relative', width: '100%' }}>
      {!googleMapsApiKey && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            backgroundColor: '#fff3cd',
            color: '#856404',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            zIndex: 10,
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
          }}
        >
          ⚠️ Development Mode (No API Key)
        </div>
      )}
      
      <GoogleMap
        mapContainerStyle={mapContainerStyle}
        center={mapCenter}
        zoom={12}
        options={mapOptions}
        onClick={handleMapClick}
      >
        {startLocation && (
          <Marker
            position={startLocation}
            label="Start"
            title={`Start Location: ${startLocation.lat.toFixed(6)}, ${startLocation.lng.toFixed(6)}`}
          />
        )}
      </GoogleMap>

      <div
        style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          backgroundColor: 'white',
          padding: '15px',
          borderRadius: '8px',
          boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
          zIndex: 1,
          minWidth: '250px',
        }}
      >
        <h3 style={{ margin: '0 0 10px 0', fontSize: '18px' }}>
          EV Traffic Prediction
        </h3>
        
        {startLocation && (
          <div style={{ marginBottom: '10px', fontSize: '12px', color: '#666' }}>
            <strong>Selected Location:</strong>
            <br />
            Lat: {startLocation.lat.toFixed(6)}
            <br />
            Lng: {startLocation.lng.toFixed(6)}
          </div>
        )}

        <button
          onClick={handlePredictEnergy}
          disabled={!startLocation || loading}
          style={{
            width: '100%',
            padding: '10px',
            backgroundColor: startLocation && !loading ? '#4285f4' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: startLocation && !loading ? 'pointer' : 'not-allowed',
            fontSize: '14px',
            fontWeight: 'bold',
            marginBottom: '10px',
          }}
        >
          {loading ? 'Calculating...' : 'Calculate Energy'}
        </button>

        <button
          onClick={handleReset}
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: '#f0f0f0',
            color: '#333',
            border: '1px solid #ddd',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '12px',
          }}
        >
          Reset
        </button>
      </div>

      {predictionResult && (
        <div
          style={{
            position: 'absolute',
            bottom: '20px',
            right: '20px',
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            zIndex: 1,
            minWidth: '300px',
            maxWidth: '400px',
          }}
        >
          <h3 style={{ margin: '0 0 15px 0', fontSize: '18px', color: '#4285f4' }}>
            Prediction Results
          </h3>

          <div style={{ marginBottom: '15px' }}>
            <div style={{ fontSize: '32px', fontWeight: 'bold', color: '#34a853' }}>
              {predictionResult.prediction.toFixed(2)} kWh
            </div>
            <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
              Predicted Energy Consumption
            </div>
          </div>

          <div style={{ borderTop: '1px solid #eee', paddingTop: '15px' }}>
            <div style={{ fontSize: '14px', marginBottom: '10px' }}>
              <strong>Route Insights:</strong>
            </div>
            
            <div style={{ fontSize: '12px', lineHeight: '1.6' }}>
              <div>
                <strong>Distance to nearest EV station:</strong>{' '}
                {(predictionResult.dist_to_nearest_ev_m / 1000).toFixed(2)} km
              </div>
              <div>
                <strong>EV stations within 500m:</strong>{' '}
                {predictionResult.ev_within_500m}
              </div>
              <div>
                <strong>Average temperature:</strong>{' '}
                {predictionResult.avg_temp.toFixed(1)}°C
              </div>
              <div>
                <strong>Total precipitation:</strong>{' '}
                {predictionResult.total_prcp.toFixed(1)} mm
              </div>
              <div>
                <strong>Road segment length:</strong>{' '}
                {predictionResult.used_SHAPE_Length.toFixed(2)} m
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div
          style={{
            position: 'absolute',
            bottom: '20px',
            left: '10px',
            backgroundColor: '#fce8e6',
            color: '#d93025',
            padding: '15px',
            borderRadius: '8px',
            boxShadow: '0 2px 6px rgba(0,0,0,0.3)',
            zIndex: 1,
            maxWidth: '300px',
          }}
        >
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(255, 255, 255, 0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2,
          }}
        >
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '18px', marginBottom: '10px' }}>
              Calculating prediction...
            </div>
            <div style={{ fontSize: '14px', color: '#666' }}>
              Please wait
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MapComponent;

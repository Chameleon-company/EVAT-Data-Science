# Google Maps Integration Guide

## Overview

This guide covers the React component (`MapComponent.tsx`) that integrates Google Maps with the EV Traffic & Weather Model API.

## Features

1. **State Management:**
   - ✅ `predictionResult` state (initially null)
   - ✅ `loading` and `error` states
   - ✅ `start_location` tracking (lat/lng from map clicks)

2. **API Integration:**
   - ✅ Imports `predictEVTraffic` from `apiClient.ts`
   - ✅ `handlePredictEnergy` function calls API with `{ year: 2026, start_lat, start_lon }`
   - ✅ Backend configured for port **5001**

3. **UI Updates:**
   - ✅ "Calculate Energy" button (visible and functional)
   - ✅ UI Card displaying prediction result (kWh) and route insights
   - ✅ Loading and error states handled gracefully

## Files

- **`MapComponent.tsx`** - Main React component with Google Maps integration
- **`apiClient.ts`** - TypeScript API client
- **`useEVPrediction.ts`** - React hook for predictions

## Setup

### Step 1: Install Dependencies

```bash
npm install @react-google-maps/api
# or
yarn add @react-google-maps/api
```

### Step 2: Get Google Maps API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/google/maps-apis)
2. Create a new project or select existing one
3. Enable "Maps JavaScript API"
4. Create credentials (API Key)
5. Add to your `.env` file:

```env
REACT_APP_GOOGLE_MAPS_API_KEY=your_api_key_here
```

### Step 3: Copy Files to Your React Project

Copy these files to your React project:
- `MapComponent.tsx`
- `apiClient.ts` (already created)
- `useEVPrediction.ts` (optional, if you prefer using the hook)

### Step 4: Use the Component

```tsx
import React from 'react';
import MapComponent from './MapComponent';

function App() {
  const apiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY || 'YOUR_KEY';
  
  return (
    <div>
      <MapComponent googleMapsApiKey={apiKey} />
    </div>
  );
}

export default App;
```

## Component Features

### Map Interaction
- Click anywhere on the map to set start location
- Marker appears at selected location
- Coordinates displayed in control panel

### Prediction Flow
1. User clicks on map → sets `start_location`
2. User clicks "Calculate Energy" button
3. API called with `{ year: 2026, start_lat, start_lon }`
4. Results displayed in card with:
   - **Prediction** (kWh) - large, prominent display
   - **Route Insights:**
     - Distance to nearest EV station
     - Number of EV stations within 500m
     - Average temperature
     - Total precipitation
     - Road segment length

### Error Handling
- Network errors displayed in error card
- Validation errors shown
- Loading states with overlay

### UI/UX
- Clean, modern design
- Responsive layout
- Loading indicators
- Error messages
- Reset functionality

## Customization

### Change Default Map Center

Edit `defaultCenter` in `MapComponent.tsx`:

```tsx
const defaultCenter = {
  lat: -37.8136,
  lng: 144.9631,
};
```

### Change Prediction Year

Edit the `handlePredictEnergy` function:

```tsx
const result = await predictEVTraffic({
  year: 2026,
  start_lat: startLocation.lat,
  start_lon: startLocation.lng,
});
```

### Customize Styling

All styles are inline and can be easily modified. The component uses:
- Absolute positioning for overlays
- Material Design-inspired colors
- Responsive sizing

## API Response Structure

The component expects this response format (from `apiClient.ts`):

```typescript
{
  year: number;
  start_lat: number;
  start_lon: number;
  dist_to_nearest_ev_m: number;
  ev_within_500m: number;
  avg_temp: number;
  total_prcp: number;
  used_SHAPE_Length: number;
  prediction: number;
}
```

## Troubleshooting

### Map Not Loading
- Check Google Maps API key is correct
- Ensure "Maps JavaScript API" is enabled
- Check browser console for errors

### API Not Responding
- Verify backend server is running on port 5001
- Check CORS settings in `api.py`
- Verify `apiClient.ts` has correct API URL

### TypeScript Errors
- Ensure `@types/react` and `@types/react-dom` are installed
- Check that `apiClient.ts` exports are correct

## Using the Hook

If you prefer using the custom hook instead:

```tsx
import { useEVPrediction } from './useEVPrediction';

function MyComponent() {
  const { predict, data, loading, error } = useEVPrediction();
  
  const handleClick = () => {
    predict({ year: 2026, start_lat: -37.8136, start_lon: 144.9631 });
  };
  
}
```

The `MapComponent.tsx` uses the direct API client, but both approaches work.

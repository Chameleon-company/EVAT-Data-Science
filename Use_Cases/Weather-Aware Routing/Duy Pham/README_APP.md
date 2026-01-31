# App Setup

## Files

- `App.tsx` - Main application component
- `App.css` - Basic styling
- `MapComponent.tsx` - Google Maps integration component
- `apiClient.ts` - API client for backend
- `useEVPrediction.ts` - React hook for predictions

## Quick Start

### Step 1: Set Up Google Maps API Key

**Option A: Using Environment Variable (Recommended)**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# REACT_APP_GOOGLE_MAPS_API_KEY=your_actual_key_here
```

**Option B: Direct in Code**
Edit `App.tsx` and replace `'YOUR_KEY_HERE'` with your actual API key:
```tsx
const GOOGLE_MAPS_API_KEY = 'your_actual_api_key_here';
```

### Step 2: Install Dependencies (if not already done)
```bash
npm install
```

### Step 3: Start the App
```bash
npm start
```

The app should open in your browser and display the MapComponent!

## Getting a Google Maps API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable "Maps JavaScript API"
4. Go to "Credentials" → "Create Credentials" → "API Key"
5. Copy the key and add it to `.env` or `App.tsx`

## Important Notes

- **Backend Server**: Make sure your Flask backend is running on port 5001
- **CORS**: The backend should have CORS enabled (already configured in `api.py`)
- **API Key Security**: Never commit your `.env` file to git!

## Usage

When you run `npm start`:
1. A header with the app title
2. A Google Map centered on Melbourne, Australia
3. A control panel on the left with:
   - Location selection info
   - "Calculate Energy" button
   - "Reset" button
4. Click anywhere on the map to select a location
5. Click "Calculate Energy" to get predictions
6. Results will appear in a card on the right

## Troubleshooting

### Map Not Loading
- Check that your Google Maps API key is correct
- Verify "Maps JavaScript API" is enabled in Google Cloud Console
- Check browser console for errors

### API Not Working
- Ensure backend is running: `python3 api.py`
- Check that backend is on port 5001
- Verify CORS is enabled in `api.py`

### TypeScript Errors
- Make sure all dependencies are installed: `npm install`
- Check that `apiClient.ts` exists and exports correctly

## Next Steps

1. Add your Google Maps API key
2. Start the backend server (`python3 api.py`)
3. Run `npm start`
4. Test the integration

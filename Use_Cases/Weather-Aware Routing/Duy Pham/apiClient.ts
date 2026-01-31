export interface PredictionRequest {
  year: number;
  start_lat: number;
  start_lon: number;
}

export interface PredictionResponse {
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

export interface ApiError {
  error: string;
  status?: number;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

export async function predictEVTraffic(params: PredictionRequest): Promise<PredictionResponse> {
  if (typeof params.year !== 'number' || !Number.isInteger(params.year)) {
    throw { error: 'Invalid year', status: 400 } as ApiError;
  }

  if (typeof params.start_lat !== 'number' || isNaN(params.start_lat) || params.start_lat < -90 || params.start_lat > 90) {
    throw { error: 'Invalid start_lat', status: 400 } as ApiError;
  }

  if (typeof params.start_lon !== 'number' || isNaN(params.start_lon) || params.start_lon < -180 || params.start_lon > 180) {
    throw { error: 'Invalid start_lon', status: 400 } as ApiError;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        year: params.year,
        start_lat: params.start_lat,
        start_lon: params.start_lon,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw { error: data.error || `HTTP error! status: ${response.status}`, status: response.status } as ApiError;
    }

    if (!data || typeof data.prediction === 'undefined') {
      throw { error: 'Invalid response format from API', status: response.status } as ApiError;
    }

    return data as PredictionResponse;
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw { error: 'Network error: Unable to connect to API. Is the server running?', status: 0 } as ApiError;
    }

    if (error && typeof error === 'object' && 'error' in error) {
      throw error as ApiError;
    }

    throw { error: `Unexpected error: ${error instanceof Error ? error.message : String(error)}`, status: 0 } as ApiError;
  }
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
      method: 'GET',
    });
    return response.ok;
  } catch {
    return false;
  }
}

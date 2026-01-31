import { useState, useCallback } from 'react';
import { predictEVTraffic, PredictionRequest, PredictionResponse, ApiError } from './apiClient';

interface UseEVPredictionReturn {
  predict: (params: PredictionRequest) => Promise<void>;
  data: PredictionResponse | null;
  loading: boolean;
  error: string | null;
  reset: () => void;
}

export function useEVPrediction(): UseEVPredictionReturn {
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = useCallback(async (params: PredictionRequest) => {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const result = await predictEVTraffic(params);
      setData(result);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.error || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    predict,
    data,
    loading,
    error,
    reset,
  };
}

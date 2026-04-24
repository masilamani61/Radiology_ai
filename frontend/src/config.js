const browserHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
const browserProtocol = typeof window !== 'undefined' ? window.location.protocol : 'http:';
const sameOrigin = typeof window !== 'undefined' ? window.location.origin : '';

export const FRONTEND_CONFIG = {
  apiUrl: process.env.REACT_APP_API_URL || sameOrigin || `${browserProtocol}//${browserHost}`,
  apiPrefix: process.env.REACT_APP_API_PREFIX || '/api/v1',
  grafanaUrl: process.env.REACT_APP_GRAFANA_URL || '',
  mlflowUrl: process.env.REACT_APP_MLFLOW_URL || '',
};

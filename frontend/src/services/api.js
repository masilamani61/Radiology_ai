import axios from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8005';

const api = axios.create({
  baseURL: `${BASE_URL}/api/v1`,
  timeout: 30000,
});

api.interceptors.response.use(
  res => res,
  err => {
    console.error('[API Error]', err.response?.data?.detail || err.message);
    return Promise.reject(err);
  }
);

export const predictXray = async (imageFile) => {
  const form = new FormData();
  form.append('file', imageFile);
  const res = await api.post('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const submitFeedback = async (data) => {
  const res = await api.post('/feedback', data);
  return res.data;
};

export const checkHealth = async () => {
  const res = await axios.get(`${BASE_URL}/health`);
  return res.data;
};

export default api;

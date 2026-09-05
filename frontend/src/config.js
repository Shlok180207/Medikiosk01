// Centralized API configuration for MediKiosk Frontend
// When running locally: 'http://localhost:8000/api'
// When running on Google Colab: replace with your Cloudflare Tunnel URL (e.g., 'https://xyz.trycloudflare.com/api')
// or set VITE_API_BASE_URL in a .env file.

export const API_BASE_URL = 
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

export default {
  API_BASE_URL
};

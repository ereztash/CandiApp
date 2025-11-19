import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor - add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor - handle errors
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config

    // If 401 and not already retried, try to refresh token
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true

      const refreshToken = localStorage.getItem('refresh_token')
      if (refreshToken) {
        try {
          const response = await axios.post(`${API_BASE_URL}/auth/refresh`, null, {
            params: { refresh_token: refreshToken }
          })

          const { access_token, refresh_token: newRefreshToken } = response.data
          localStorage.setItem('access_token', access_token)
          localStorage.setItem('refresh_token', newRefreshToken)

          originalRequest.headers.Authorization = `Bearer ${access_token}`
          return api(originalRequest)
        } catch (refreshError) {
          // Refresh failed, logout
          localStorage.removeItem('access_token')
          localStorage.removeItem('refresh_token')
          window.location.href = '/login'
        }
      }
    }

    return Promise.reject(error)
  }
)

// Auth API
export const authAPI = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => api.post('/auth/login', data),
  getMe: () => api.get('/auth/me'),
  refresh: (refreshToken) => api.post('/auth/refresh', null, {
    params: { refresh_token: refreshToken }
  }),
}

// Resumes API
export const resumesAPI = {
  parse: (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/resumes/parse', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  list: (skip = 0, limit = 20) => api.get('/resumes', { params: { skip, limit } }),
  get: (id) => api.get(`/resumes/${id}`),
  delete: (id) => api.delete(`/resumes/${id}`),
}

// Scoring API
export const scoringAPI = {
  score: (resumeId, jobRequirements) => api.post('/score', {
    resume_id: resumeId,
    job_requirements: jobRequirements,
  }),
  batchScore: (resumeIds, jobRequirements) => api.post('/score/batch', {
    resume_ids: resumeIds,
    job_requirements: jobRequirements,
  }),
  getScores: (resumeId) => api.get(`/scores/${resumeId}`),
}

// Health API
export const healthAPI = {
  check: () => api.get('/health'),
  features: () => api.get('/features'),
}

export default api

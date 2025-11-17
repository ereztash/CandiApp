/**
 * API Service
 * Handles all API calls to the backend
 */
import axios, { AxiosInstance } from 'axios'
import type {
  Job,
  JobCreate,
  JobWithStats,
  Candidate,
  ScreeningRequest,
  ScreeningResponse,
  ScreeningResult,
  ScreeningDecision,
  ProcessArchetype,
  CandidateWithResult,
} from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const API_V1_PREFIX = '/api/v1'

class ApiService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}${API_V1_PREFIX}`,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 seconds
    })

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error)
        return Promise.reject(error)
      }
    )
  }

  // ===== Jobs =====

  async createJob(job: JobCreate): Promise<Job> {
    const response = await this.client.post<Job>('/jobs/', job)
    return response.data
  }

  async listJobs(activeOnly: boolean = true): Promise<Job[]> {
    const response = await this.client.get<Job[]>('/jobs/', {
      params: { active_only: activeOnly },
    })
    return response.data
  }

  async getJob(jobId: string): Promise<Job> {
    const response = await this.client.get<Job>(`/jobs/${jobId}`)
    return response.data
  }

  async updateJob(jobId: string, updates: Partial<JobCreate>): Promise<Job> {
    const response = await this.client.put<Job>(`/jobs/${jobId}`, updates)
    return response.data
  }

  async deleteJob(jobId: string, hardDelete: boolean = false): Promise<void> {
    await this.client.delete(`/jobs/${jobId}`, {
      params: { hard_delete: hardDelete },
    })
  }

  async getJobStats(jobId: string): Promise<JobWithStats> {
    const response = await this.client.get<JobWithStats>(`/jobs/${jobId}/stats`)
    return response.data
  }

  // ===== Screening =====

  async screenCandidate(request: ScreeningRequest): Promise<ScreeningResponse> {
    const response = await this.client.post<ScreeningResponse>(
      '/screening/screen',
      request
    )
    return response.data
  }

  async getScreeningResult(candidateId: string): Promise<ScreeningResult> {
    const response = await this.client.get<ScreeningResult>(
      `/screening/${candidateId}/result`
    )
    return response.data
  }

  // ===== Candidates =====

  async listCandidates(jobId?: string): Promise<Candidate[]> {
    const response = await this.client.get<Candidate[]>('/candidates/', {
      params: jobId ? { job_id: jobId } : {},
    })
    return response.data
  }

  async getCandidate(candidateId: string): Promise<Candidate> {
    const response = await this.client.get<Candidate>(
      `/candidates/${candidateId}`
    )
    return response.data
  }

  async getCandidateScreening(candidateId: string): Promise<ScreeningResult> {
    const response = await this.client.get<ScreeningResult>(
      `/candidates/${candidateId}/screening`
    )
    return response.data
  }

  async searchByDecision(
    decision: ScreeningDecision,
    jobId?: string
  ): Promise<CandidateWithResult[]> {
    const response = await this.client.get<CandidateWithResult[]>(
      '/candidates/search/by-decision',
      {
        params: {
          decision,
          job_id: jobId,
        },
      }
    )
    return response.data
  }

  async searchByArchetype(
    archetype: ProcessArchetype,
    jobId?: string
  ): Promise<CandidateWithResult[]> {
    const response = await this.client.get<CandidateWithResult[]>(
      '/candidates/search/by-archetype',
      {
        params: {
          archetype,
          job_id: jobId,
        },
      }
    )
    return response.data
  }

  // ===== Health =====

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await this.client.get('/health/')
    return response.data
  }

  async databaseHealth(): Promise<{ status: string; database: string }> {
    const response = await this.client.get('/health/db')
    return response.data
  }
}

// Export singleton instance
export const apiService = new ApiService()
export default apiService

/**
 * Type definitions for the Resume Screening System
 */

export enum ProcessArchetype {
  INNOVATOR = 'Innovator',
  LEADER = 'Leader',
  MAINTAINER = 'Maintainer',
  PROBLEM_SOLVER = 'Problem-Solver',
  ENABLER = 'Enabler',
}

export enum ScreeningDecision {
  PASSED = 'PASSED',
  FAILED = 'FAILED',
  PENDING_REVIEW = 'PENDING_REVIEW',
}

export enum RejectionReason {
  ARCHETYPE_MISMATCH = 'ARCHETYPE_MISMATCH',
  INSUFFICIENT_SKILLS = 'INSUFFICIENT_SKILLS',
  LOW_PROCESS_FIT = 'LOW_PROCESS_FIT',
  LOW_SEMANTIC_FIT = 'LOW_SEMANTIC_FIT',
  LOW_OVERALL_SCORE = 'LOW_OVERALL_SCORE',
  INSUFFICIENT_EXPERIENCE = 'INSUFFICIENT_EXPERIENCE',
}

export interface Job {
  id: string
  name: string
  description?: string
  archetype_primary: ProcessArchetype
  archetype_secondary?: ProcessArchetype
  required_skills: string[]
  preferred_skills: string[]
  min_experience_years?: number
  is_active: boolean
  total_candidates_screened: number
  created_at: string
  updated_at: string
}

export interface JobCreate {
  name: string
  description?: string
  archetype_primary: ProcessArchetype
  archetype_secondary?: ProcessArchetype
  required_skills: string[]
  preferred_skills: string[]
  min_experience_years?: number
}

export interface JobWithStats extends Job {
  total_screened: number
  total_passed: number
  total_failed: number
  pass_rate: number
  avg_overall_score: number
  avg_process_fit: number
  avg_semantic_fit: number
  archetype_distribution: Record<string, number>
}

export interface Candidate {
  id: string
  name: string
  email: string
  resume_text: string
  job_id: string
  created_at: string
}

export interface ScreeningResult {
  id: string
  candidate_id: string
  job_id: string
  decision: ScreeningDecision
  overall_score: number
  process_fit_score: number
  semantic_fit_score: number
  archetype_detected: ProcessArchetype
  archetype_confidence: number
  archetype_alignment: string
  evidence_verbs: string[]
  matched_skills: string[]
  missing_skills: string[]
  rejection_reason?: RejectionReason
  rejection_details?: string
  recommendation: string
  email_sent: boolean
  hr_reviewed: boolean
  hr_review_timestamp?: string
  hr_notes?: string
  screened_at: string
}

export interface ScreeningRequest {
  candidate_name: string
  candidate_email: string
  resume_text: string
  job_id: string
}

export interface ScreeningResponse {
  success: boolean
  screening_result: ScreeningResult
  candidate_id: string
  message: string
}

export interface CandidateWithResult {
  candidate: Candidate
  screening_result: ScreeningResult
}

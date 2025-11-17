/**
 * ScreeningPage Component
 * Main page for screening resumes
 */
import React, { useState, useEffect } from 'react'
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
} from '@mui/material'
import { Upload, Search } from '@mui/icons-material'
import { useDropzone } from 'react-dropzone'
import apiService from '../services/api'
import ScreeningResult from '../components/ScreeningResult'
import type {
  Job,
  ScreeningRequest,
  ScreeningResult as ScreeningResultType,
} from '../types'

export const ScreeningPage: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string>('')
  const [candidateName, setCandidateName] = useState('')
  const [candidateEmail, setCandidateEmail] = useState('')
  const [resumeText, setResumeText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ScreeningResultType | null>(null)

  // Load jobs on mount
  useEffect(() => {
    loadJobs()
  }, [])

  const loadJobs = async () => {
    try {
      const jobsList = await apiService.listJobs(true)
      setJobs(jobsList)
      if (jobsList.length > 0 && !selectedJobId) {
        setSelectedJobId(jobsList[0].id)
      }
    } catch (err) {
      console.error('Failed to load jobs:', err)
      setError('Failed to load jobs')
    }
  }

  // File drop handler
  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      const reader = new FileReader()

      reader.onload = (e) => {
        const text = e.target?.result as string
        setResumeText(text)
      }

      reader.readAsText(file)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.txt'],
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    maxFiles: 1,
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setResult(null)

    // Validation
    if (!selectedJobId) {
      setError('Please select a job')
      return
    }
    if (!candidateName.trim()) {
      setError('Please enter candidate name')
      return
    }
    if (!candidateEmail.trim()) {
      setError('Please enter candidate email')
      return
    }
    if (!resumeText.trim()) {
      setError('Please provide resume text')
      return
    }

    setIsLoading(true)

    try {
      const request: ScreeningRequest = {
        candidate_name: candidateName,
        candidate_email: candidateEmail,
        resume_text: resumeText,
        job_id: selectedJobId,
      }

      const response = await apiService.screenCandidate(request)
      setResult(response.screening_result)
    } catch (err: any) {
      console.error('Screening failed:', err)
      setError(
        err.response?.data?.detail || 'Screening failed. Please try again.'
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setCandidateName('')
    setCandidateEmail('')
    setResumeText('')
    setResult(null)
    setError(null)
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom fontWeight="bold">
        Resume Screening
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Upload a resume and screen it against job requirements using Process-Fit
        analysis
      </Typography>

      <Box display="flex" gap={3} flexDirection={{ xs: 'column', md: 'row' }}>
        {/* Left side - Input form */}
        <Paper elevation={2} sx={{ flex: 1, p: 3 }}>
          <form onSubmit={handleSubmit}>
            {/* Job Selection */}
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Select Job</InputLabel>
              <Select
                value={selectedJobId}
                onChange={(e) => setSelectedJobId(e.target.value)}
                label="Select Job"
                required
              >
                {jobs.map((job) => (
                  <MenuItem key={job.id} value={job.id}>
                    {job.name} - {job.archetype_primary}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Candidate Info */}
            <TextField
              fullWidth
              label="Candidate Name"
              value={candidateName}
              onChange={(e) => setCandidateName(e.target.value)}
              required
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              label="Candidate Email"
              type="email"
              value={candidateEmail}
              onChange={(e) => setCandidateEmail(e.target.value)}
              required
              sx={{ mb: 3 }}
            />

            {/* Resume Upload */}
            <Box
              {...getRootProps()}
              sx={{
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'grey.300',
                borderRadius: 2,
                p: 3,
                mb: 2,
                textAlign: 'center',
                cursor: 'pointer',
                backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
                transition: 'all 0.2s',
                '&:hover': {
                  borderColor: 'primary.main',
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <input {...getInputProps()} />
              <Upload sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
              <Typography variant="body1">
                {isDragActive
                  ? 'Drop the resume file here'
                  : 'Drag & drop resume file, or click to select'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Supports .txt, .pdf, .doc, .docx
              </Typography>
            </Box>

            {/* Resume Text */}
            <TextField
              fullWidth
              label="Resume Text"
              multiline
              rows={10}
              value={resumeText}
              onChange={(e) => setResumeText(e.target.value)}
              placeholder="Paste resume text here or upload a file above"
              required
              sx={{ mb: 3 }}
            />

            {/* Error Display */}
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            {/* Submit Buttons */}
            <Box display="flex" gap={2}>
              <Button
                type="submit"
                variant="contained"
                size="large"
                startIcon={isLoading ? <CircularProgress size={20} /> : <Search />}
                disabled={isLoading}
                fullWidth
              >
                {isLoading ? 'Screening...' : 'Screen Resume'}
              </Button>
              <Button
                type="button"
                variant="outlined"
                size="large"
                onClick={handleReset}
                disabled={isLoading}
              >
                Reset
              </Button>
            </Box>
          </form>
        </Paper>

        {/* Right side - Results */}
        {result && (
          <Box flex={1}>
            <ScreeningResult result={result} />
          </Box>
        )}
      </Box>
    </Container>
  )
}

export default ScreeningPage

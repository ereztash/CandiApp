/**
 * ResultsPage Component
 * Displays screening results for all candidates
 */
import React, { useState, useEffect } from 'react'
import {
  Container,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
} from '@mui/material'
import { CheckCircle, Cancel, Visibility } from '@mui/icons-material'
import { format } from 'date-fns'
import apiService from '../services/api'
import ScreeningResult from '../components/ScreeningResult'
import type {
  Job,
  Candidate,
  ScreeningResult as ScreeningResultType,
  ScreeningDecision,
  ProcessArchetype,
} from '../types'

interface CandidateRow {
  candidate: Candidate
  result: ScreeningResultType
}

export const ResultsPage: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string>('all')
  const [filterDecision, setFilterDecision] = useState<string>('all')
  const [candidates, setCandidates] = useState<CandidateRow[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedResult, setSelectedResult] = useState<ScreeningResultType | null>(
    null
  )
  const [dialogOpen, setDialogOpen] = useState(false)

  useEffect(() => {
    loadJobs()
  }, [])

  useEffect(() => {
    loadCandidates()
  }, [selectedJobId, filterDecision])

  const loadJobs = async () => {
    try {
      const jobsList = await apiService.listJobs(true)
      setJobs(jobsList)
    } catch (err) {
      console.error('Failed to load jobs:', err)
      setError('Failed to load jobs')
    }
  }

  const loadCandidates = async () => {
    setIsLoading(true)
    setError(null)

    try {
      // Get all candidates for the selected job
      const jobId = selectedJobId === 'all' ? undefined : selectedJobId
      const candidatesList = await apiService.listCandidates(jobId)

      // Get screening results for each candidate
      const rows: CandidateRow[] = []
      for (const candidate of candidatesList) {
        try {
          const result = await apiService.getCandidateScreening(candidate.id)

          // Apply filter
          if (
            filterDecision === 'all' ||
            result.decision === filterDecision
          ) {
            rows.push({ candidate, result })
          }
        } catch (err) {
          console.error(
            `Failed to load result for candidate ${candidate.id}:`,
            err
          )
        }
      }

      setCandidates(rows)
    } catch (err) {
      console.error('Failed to load candidates:', err)
      setError('Failed to load candidates')
    } finally {
      setIsLoading(false)
    }
  }

  const handleViewDetails = (result: ScreeningResultType) => {
    setSelectedResult(result)
    setDialogOpen(true)
  }

  const getDecisionChip = (decision: ScreeningDecision) => {
    switch (decision) {
      case 'PASSED':
        return (
          <Chip
            icon={<CheckCircle />}
            label="Passed"
            color="success"
            size="small"
          />
        )
      case 'FAILED':
        return (
          <Chip icon={<Cancel />} label="Failed" color="error" size="small" />
        )
      default:
        return (
          <Chip label="Pending" color="warning" size="small" />
        )
    }
  }

  const getArchetypeColor = (archetype: ProcessArchetype): string => {
    const colors: Record<ProcessArchetype, string> = {
      'Innovator': 'secondary',
      'Leader': 'primary',
      'Maintainer': 'success',
      'Problem-Solver': 'warning',
      'Enabler': 'info',
    }
    return colors[archetype] || 'default'
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom fontWeight="bold">
        Screening Results
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        View and manage all candidate screening results
      </Typography>

      {/* Filters */}
      <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
        <Box display="flex" gap={2}>
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Job</InputLabel>
            <Select
              value={selectedJobId}
              onChange={(e) => setSelectedJobId(e.target.value)}
              label="Job"
            >
              <MenuItem value="all">All Jobs</MenuItem>
              {jobs.map((job) => (
                <MenuItem key={job.id} value={job.id}>
                  {job.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Decision</InputLabel>
            <Select
              value={filterDecision}
              onChange={(e) => setFilterDecision(e.target.value)}
              label="Decision"
            >
              <MenuItem value="all">All Decisions</MenuItem>
              <MenuItem value="PASSED">Passed</MenuItem>
              <MenuItem value="FAILED">Failed</MenuItem>
              <MenuItem value="PENDING_REVIEW">Pending Review</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Paper>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Loading State */}
      {isLoading && (
        <Box display="flex" justifyContent="center" py={4}>
          <CircularProgress />
        </Box>
      )}

      {/* Results Table */}
      {!isLoading && candidates.length > 0 && (
        <TableContainer component={Paper} elevation={2}>
          <Table>
            <TableHead>
              <TableRow sx={{ backgroundColor: 'grey.100' }}>
                <TableCell><strong>Candidate</strong></TableCell>
                <TableCell><strong>Email</strong></TableCell>
                <TableCell><strong>Decision</strong></TableCell>
                <TableCell><strong>Overall Score</strong></TableCell>
                <TableCell><strong>Archetype</strong></TableCell>
                <TableCell><strong>Alignment</strong></TableCell>
                <TableCell><strong>Screened At</strong></TableCell>
                <TableCell align="center"><strong>Actions</strong></TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {candidates.map(({ candidate, result }) => (
                <TableRow key={candidate.id} hover>
                  <TableCell>{candidate.name}</TableCell>
                  <TableCell>{candidate.email}</TableCell>
                  <TableCell>{getDecisionChip(result.decision)}</TableCell>
                  <TableCell>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="body2" fontWeight="bold">
                        {Math.round(result.overall_score * 100)}%
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={result.archetype_detected}
                      color={getArchetypeColor(result.archetype_detected) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography
                      variant="caption"
                      sx={{
                        color:
                          result.archetype_alignment === 'perfect'
                            ? 'success.main'
                            : result.archetype_alignment === 'mismatch'
                            ? 'error.main'
                            : 'text.secondary',
                      }}
                    >
                      {result.archetype_alignment}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="caption">
                      {format(new Date(result.screened_at), 'MMM d, yyyy HH:mm')}
                    </Typography>
                  </TableCell>
                  <TableCell align="center">
                    <IconButton
                      size="small"
                      color="primary"
                      onClick={() => handleViewDetails(result)}
                    >
                      <Visibility />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Empty State */}
      {!isLoading && candidates.length === 0 && (
        <Paper elevation={2} sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary">
            No candidates found
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Try adjusting your filters or screen some candidates first
          </Typography>
        </Paper>
      )}

      {/* Details Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Screening Details</DialogTitle>
        <DialogContent>
          {selectedResult && <ScreeningResult result={selectedResult} />}
        </DialogContent>
      </Dialog>
    </Container>
  )
}

export default ResultsPage

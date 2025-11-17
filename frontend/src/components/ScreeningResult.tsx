/**
 * ScreeningResult Component
 * Displays the result of a resume screening
 */
import React from 'react'
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
  Button,
  Alert,
} from '@mui/material'
import {
  CheckCircle,
  Cancel,
  Psychology,
  School,
  EmojiEvents,
} from '@mui/icons-material'
import type { ScreeningResult as ScreeningResultType, ProcessArchetype } from '../types'

interface ScreeningResultProps {
  result: ScreeningResultType
}

const ARCHETYPE_COLORS: Record<ProcessArchetype, string> = {
  'Innovator': '#9c27b0',
  'Leader': '#1976d2',
  'Maintainer': '#2e7d32',
  'Problem-Solver': '#ed6c02',
  'Enabler': '#0288d1',
}

const ARCHETYPE_ICONS: Record<ProcessArchetype, React.ReactNode> = {
  'Innovator': <EmojiEvents />,
  'Leader': <Psychology />,
  'Maintainer': <School />,
  'Problem-Solver': <Psychology />,
  'Enabler': <School />,
}

export const ScreeningResult: React.FC<ScreeningResultProps> = ({ result }) => {
  const isPassed = result.decision === 'PASSED'
  const scorePercentage = Math.round(result.overall_score * 100)
  const archetypeColor = ARCHETYPE_COLORS[result.archetype_detected]

  const copyToClipboard = () => {
    const text = `
Screening Result
================
Decision: ${result.decision}
Overall Score: ${scorePercentage}%
Process Fit: ${Math.round(result.process_fit_score * 100)}%
Semantic Fit: ${Math.round(result.semantic_fit_score * 100)}%

Archetype: ${result.archetype_detected} (${Math.round(result.archetype_confidence * 100)}%)
Alignment: ${result.archetype_alignment}

Matched Skills: ${result.matched_skills.join(', ')}
Missing Skills: ${result.missing_skills.join(', ')}

Recommendation: ${result.recommendation}
    `.trim()

    navigator.clipboard.writeText(text)
    alert('Copied to clipboard!')
  }

  return (
    <Card elevation={3}>
      <CardContent>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            {isPassed ? (
              <CheckCircle color="success" fontSize="large" />
            ) : (
              <Cancel color="error" fontSize="large" />
            )}
            <Typography variant="h5" fontWeight="bold">
              {isPassed ? 'PASSED' : 'FAILED'}
            </Typography>
          </Box>
          <Button variant="outlined" size="small" onClick={copyToClipboard}>
            Copy Result
          </Button>
        </Box>

        {/* Overall Score */}
        <Box mb={3}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Overall Score
          </Typography>
          <Box display="flex" alignItems="center" gap={2}>
            <Box flex={1}>
              <LinearProgress
                variant="determinate"
                value={scorePercentage}
                sx={{
                  height: 10,
                  borderRadius: 5,
                  backgroundColor: '#e0e0e0',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: isPassed ? '#4caf50' : '#f44336',
                  },
                }}
              />
            </Box>
            <Typography variant="h6" fontWeight="bold">
              {scorePercentage}%
            </Typography>
          </Box>
        </Box>

        {/* Sub-scores */}
        <Box display="flex" gap={2} mb={3}>
          <Box flex={1}>
            <Typography variant="caption" color="text.secondary">
              Process Fit
            </Typography>
            <Typography variant="h6">
              {Math.round(result.process_fit_score * 100)}%
            </Typography>
          </Box>
          <Box flex={1}>
            <Typography variant="caption" color="text.secondary">
              Semantic Fit
            </Typography>
            <Typography variant="h6">
              {Math.round(result.semantic_fit_score * 100)}%
            </Typography>
          </Box>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Archetype */}
        <Box mb={3}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Detected Archetype
          </Typography>
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            <Chip
              label={result.archetype_detected}
              sx={{
                backgroundColor: archetypeColor,
                color: 'white',
                fontWeight: 'bold',
              }}
              icon={
                <Box sx={{ color: 'white !important' }}>
                  {ARCHETYPE_ICONS[result.archetype_detected]}
                </Box>
              }
            />
            <Typography variant="body2" color="text.secondary">
              {Math.round(result.archetype_confidence * 100)}% confidence
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary">
            Alignment: <strong>{result.archetype_alignment}</strong>
          </Typography>
        </Box>

        {/* Evidence Verbs */}
        {result.evidence_verbs.length > 0 && (
          <Box mb={3}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Evidence Verbs
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={0.5}>
              {result.evidence_verbs.map((verb, index) => (
                <Chip key={index} label={verb} size="small" variant="outlined" />
              ))}
            </Box>
          </Box>
        )}

        <Divider sx={{ my: 2 }} />

        {/* Skills */}
        <Box mb={3}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Matched Skills ({result.matched_skills.length})
          </Typography>
          <Box display="flex" flexWrap="wrap" gap={0.5} mb={2}>
            {result.matched_skills.length > 0 ? (
              result.matched_skills.map((skill, index) => (
                <Chip
                  key={index}
                  label={skill}
                  size="small"
                  color="success"
                  variant="outlined"
                />
              ))
            ) : (
              <Typography variant="body2" color="text.secondary">
                No skills matched
              </Typography>
            )}
          </Box>

          {result.missing_skills.length > 0 && (
            <>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Missing Skills ({result.missing_skills.length})
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={0.5}>
                {result.missing_skills.map((skill, index) => (
                  <Chip
                    key={index}
                    label={skill}
                    size="small"
                    color="error"
                    variant="outlined"
                  />
                ))}
              </Box>
            </>
          )}
        </Box>

        {/* Recommendation */}
        <Alert severity={isPassed ? 'success' : 'info'} sx={{ mt: 2 }}>
          <Typography variant="body2">
            <strong>Recommendation:</strong> {result.recommendation}
          </Typography>
        </Alert>

        {/* Rejection Details */}
        {!isPassed && result.rejection_details && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>Reason:</strong> {result.rejection_details}
            </Typography>
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}

export default ScreeningResult

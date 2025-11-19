import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { resumesAPI, scoringAPI } from '../services/api'
import toast from 'react-hot-toast'
import {
  PlusIcon,
  XMarkIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline'

export default function Score() {
  const [searchParams] = useSearchParams()
  const [resumes, setResumes] = useState([])
  const [selectedResume, setSelectedResume] = useState(searchParams.get('resume') || '')
  const [loading, setLoading] = useState(true)
  const [scoring, setScoring] = useState(false)
  const [result, setResult] = useState(null)

  // Job requirements
  const [requiredSkills, setRequiredSkills] = useState([])
  const [preferredSkills, setPreferredSkills] = useState([])
  const [minYears, setMinYears] = useState('')
  const [maxYears, setMaxYears] = useState('')
  const [jobTitle, setJobTitle] = useState('')

  // Input states
  const [newRequiredSkill, setNewRequiredSkill] = useState('')
  const [newPreferredSkill, setNewPreferredSkill] = useState('')

  useEffect(() => {
    fetchResumes()
  }, [])

  const fetchResumes = async () => {
    try {
      const response = await resumesAPI.list(0, 100)
      setResumes(response.data)
    } catch (error) {
      toast.error('Failed to load resumes')
    } finally {
      setLoading(false)
    }
  }

  const addRequiredSkill = () => {
    if (newRequiredSkill.trim()) {
      setRequiredSkills([...requiredSkills, newRequiredSkill.trim()])
      setNewRequiredSkill('')
    }
  }

  const addPreferredSkill = () => {
    if (newPreferredSkill.trim()) {
      setPreferredSkills([...preferredSkills, newPreferredSkill.trim()])
      setNewPreferredSkill('')
    }
  }

  const removeSkill = (list, setList, index) => {
    setList(list.filter((_, i) => i !== index))
  }

  const handleScore = async () => {
    if (!selectedResume) {
      toast.error('Please select a resume')
      return
    }

    setScoring(true)
    setResult(null)

    try {
      const response = await scoringAPI.score(selectedResume, {
        required_skills: requiredSkills,
        preferred_skills: preferredSkills,
        min_years_experience: minYears ? parseInt(minYears) : null,
        max_years_experience: maxYears ? parseInt(maxYears) : null,
        job_title: jobTitle || null,
      })

      setResult(response.data)
      toast.success('Resume scored successfully!')
    } catch (error) {
      const message = error.response?.data?.message || 'Failed to score resume'
      toast.error(message)
    } finally {
      setScoring(false)
    }
  }

  const getScoreColor = (score) => {
    if (score >= 75) return 'text-green-600'
    if (score >= 50) return 'text-yellow-600'
    return 'text-red-600'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-8">Score Resume</h1>

      <div className="grid grid-cols-2 gap-6">
        {/* Input form */}
        <div className="space-y-6">
          {/* Resume selection */}
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Select Resume</h2>
            <select
              value={selectedResume}
              onChange={(e) => setSelectedResume(e.target.value)}
              className="input-field"
            >
              <option value="">Choose a resume...</option>
              {resumes.map((resume) => (
                <option key={resume.id} value={resume.id}>
                  {resume.parsed_data?.personal?.full_name || resume.file_name}
                </option>
              ))}
            </select>
          </div>

          {/* Job requirements */}
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Job Requirements</h2>

            <div className="space-y-4">
              {/* Job title */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Job Title
                </label>
                <input
                  type="text"
                  value={jobTitle}
                  onChange={(e) => setJobTitle(e.target.value)}
                  className="input-field"
                  placeholder="Senior Software Engineer"
                />
              </div>

              {/* Experience range */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Min Years
                  </label>
                  <input
                    type="number"
                    value={minYears}
                    onChange={(e) => setMinYears(e.target.value)}
                    className="input-field"
                    min="0"
                    placeholder="0"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Max Years
                  </label>
                  <input
                    type="number"
                    value={maxYears}
                    onChange={(e) => setMaxYears(e.target.value)}
                    className="input-field"
                    min="0"
                    placeholder="10"
                  />
                </div>
              </div>

              {/* Required skills */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Required Skills
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newRequiredSkill}
                    onChange={(e) => setNewRequiredSkill(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && addRequiredSkill()}
                    className="input-field flex-1"
                    placeholder="Add skill..."
                  />
                  <button onClick={addRequiredSkill} className="btn-secondary px-3">
                    <PlusIcon className="w-5 h-5" />
                  </button>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {requiredSkills.map((skill, idx) => (
                    <span
                      key={idx}
                      className="inline-flex items-center px-2 py-1 bg-red-100 text-red-700 rounded text-sm"
                    >
                      {skill}
                      <button
                        onClick={() => removeSkill(requiredSkills, setRequiredSkills, idx)}
                        className="ml-1"
                      >
                        <XMarkIcon className="w-4 h-4" />
                      </button>
                    </span>
                  ))}
                </div>
              </div>

              {/* Preferred skills */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Preferred Skills
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newPreferredSkill}
                    onChange={(e) => setNewPreferredSkill(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && addPreferredSkill()}
                    className="input-field flex-1"
                    placeholder="Add skill..."
                  />
                  <button onClick={addPreferredSkill} className="btn-secondary px-3">
                    <PlusIcon className="w-5 h-5" />
                  </button>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {preferredSkills.map((skill, idx) => (
                    <span
                      key={idx}
                      className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-700 rounded text-sm"
                    >
                      {skill}
                      <button
                        onClick={() => removeSkill(preferredSkills, setPreferredSkills, idx)}
                        className="ml-1"
                      >
                        <XMarkIcon className="w-4 h-4" />
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <button
              onClick={handleScore}
              disabled={scoring || !selectedResume}
              className="mt-6 w-full btn-primary py-3 disabled:opacity-50"
            >
              {scoring ? 'Scoring...' : 'Score Resume'}
            </button>
          </div>
        </div>

        {/* Results */}
        <div>
          {result ? (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Scoring Result</h2>

              {/* Overall score */}
              <div className="text-center mb-6">
                <div className={`text-5xl font-bold ${getScoreColor(result.result.overall_score)}`}>
                  {result.result.overall_score.toFixed(0)}%
                </div>
                <p className="text-gray-500 mt-1">{result.result.ranking}</p>
              </div>

              {/* Dimension scores */}
              <div className="space-y-3 mb-6">
                {Object.entries(result.result.dimension_scores).map(([key, value]) => (
                  <div key={key}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                      <span className={`font-medium ${getScoreColor(value)}`}>
                        {value.toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          value >= 75 ? 'bg-green-500' :
                          value >= 50 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${value}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Match details */}
              {result.result.match_details && (
                <div className="border-t pt-4">
                  <h3 className="font-medium mb-3">Match Details</h3>

                  {result.result.match_details.matched_required_skills?.length > 0 && (
                    <div className="mb-3">
                      <p className="text-sm text-gray-500 mb-1">Matched Skills</p>
                      <div className="flex flex-wrap gap-1">
                        {result.result.match_details.matched_required_skills.map((skill, idx) => (
                          <span key={idx} className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs">
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {result.result.match_details.missing_required_skills?.length > 0 && (
                    <div className="mb-3">
                      <p className="text-sm text-gray-500 mb-1">Missing Skills</p>
                      <div className="flex flex-wrap gap-1">
                        {result.result.match_details.missing_required_skills.map((skill, idx) => (
                          <span key={idx} className="px-2 py-0.5 bg-red-100 text-red-700 rounded text-xs">
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Recommendations */}
              {result.result.recommendations?.length > 0 && (
                <div className="border-t pt-4 mt-4">
                  <h3 className="font-medium mb-2">Recommendations</h3>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {result.result.recommendations.map((rec, idx) => (
                      <li key={idx}>• {rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="card text-center py-12">
              <ChartBarIcon className="w-12 h-12 mx-auto text-gray-300 mb-4" />
              <p className="text-gray-500">
                Select a resume and define job requirements to see the score
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

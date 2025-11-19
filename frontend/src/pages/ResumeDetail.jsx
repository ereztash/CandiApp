import { useState, useEffect } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { resumesAPI, scoringAPI } from '../services/api'
import toast from 'react-hot-toast'
import {
  ArrowLeftIcon,
  EnvelopeIcon,
  PhoneIcon,
  LinkIcon,
  BriefcaseIcon,
  AcademicCapIcon,
} from '@heroicons/react/24/outline'

export default function ResumeDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [resume, setResume] = useState(null)
  const [scores, setScores] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchResumeData()
  }, [id])

  const fetchResumeData = async () => {
    try {
      const [resumeRes, scoresRes] = await Promise.all([
        resumesAPI.get(id),
        scoringAPI.getScores(id).catch(() => ({ data: [] })),
      ])
      setResume(resumeRes.data)
      setScores(scoresRes.data)
    } catch (error) {
      toast.error('Failed to load resume')
      navigate('/resumes')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (!resume) return null

  const { parsed_data } = resume

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center">
          <button
            onClick={() => navigate('/resumes')}
            className="mr-4 p-2 hover:bg-gray-100 rounded-lg"
          >
            <ArrowLeftIcon className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              {parsed_data?.personal?.full_name || 'Unknown Name'}
            </h1>
            <p className="text-gray-500">{resume.file_name}</p>
          </div>
        </div>
        <Link to={`/score?resume=${id}`} className="btn-primary">
          Score Resume
        </Link>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Main content */}
        <div className="col-span-2 space-y-6">
          {/* Contact */}
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Contact Information</h2>
            <div className="grid grid-cols-2 gap-4">
              {parsed_data?.contact?.email && (
                <div className="flex items-center">
                  <EnvelopeIcon className="w-5 h-5 text-gray-400 mr-2" />
                  <span>{parsed_data.contact.email}</span>
                </div>
              )}
              {parsed_data?.contact?.phone && (
                <div className="flex items-center">
                  <PhoneIcon className="w-5 h-5 text-gray-400 mr-2" />
                  <span>{parsed_data.contact.phone}</span>
                </div>
              )}
              {parsed_data?.contact?.linkedin && (
                <div className="flex items-center">
                  <LinkIcon className="w-5 h-5 text-gray-400 mr-2" />
                  <a href={parsed_data.contact.linkedin} target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline">
                    LinkedIn
                  </a>
                </div>
              )}
              {parsed_data?.contact?.github && (
                <div className="flex items-center">
                  <LinkIcon className="w-5 h-5 text-gray-400 mr-2" />
                  <a href={parsed_data.contact.github} target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:underline">
                    GitHub
                  </a>
                </div>
              )}
            </div>
          </div>

          {/* Summary */}
          {parsed_data?.summary && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Summary</h2>
              <p className="text-gray-700 whitespace-pre-line">{parsed_data.summary}</p>
            </div>
          )}

          {/* Experience */}
          {parsed_data?.experiences?.length > 0 && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Experience</h2>
              <div className="space-y-6">
                {parsed_data.experiences.map((exp, idx) => (
                  <div key={idx} className="border-l-2 border-primary-200 pl-4">
                    <div className="flex items-start">
                      <BriefcaseIcon className="w-5 h-5 text-primary-500 mr-2 mt-0.5" />
                      <div>
                        <p className="font-medium">{exp.title}</p>
                        <p className="text-gray-600">{exp.company}</p>
                        {exp.location && (
                          <p className="text-sm text-gray-500">{exp.location}</p>
                        )}
                        {exp.description && (
                          <p className="mt-2 text-sm text-gray-700">{exp.description}</p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Education */}
          {parsed_data?.education?.length > 0 && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Education</h2>
              <div className="space-y-4">
                {parsed_data.education.map((edu, idx) => (
                  <div key={idx} className="flex items-start">
                    <AcademicCapIcon className="w-5 h-5 text-primary-500 mr-2 mt-0.5" />
                    <div>
                      <p className="font-medium">{edu.degree || 'Degree'}</p>
                      <p className="text-gray-600">{edu.institution}</p>
                      {edu.field_of_study && (
                        <p className="text-sm text-gray-500">{edu.field_of_study}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Meta */}
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Details</h2>
            <div className="space-y-3 text-sm">
              <div>
                <p className="text-gray-500">Total Experience</p>
                <p className="font-medium">
                  {parsed_data?.total_experience_years
                    ? `${parsed_data.total_experience_years} years`
                    : 'Not detected'}
                </p>
              </div>
              <div>
                <p className="text-gray-500">Parsing Time</p>
                <p className="font-medium">
                  {resume.parsing_time
                    ? `${(resume.parsing_time * 1000).toFixed(0)}ms`
                    : 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-gray-500">Uploaded</p>
                <p className="font-medium">
                  {new Date(resume.created_at).toLocaleString()}
                </p>
              </div>
            </div>
          </div>

          {/* Skills */}
          {parsed_data?.technical_skills?.length > 0 && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Technical Skills</h2>
              <div className="flex flex-wrap gap-2">
                {parsed_data.technical_skills.map((skill, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-1 bg-primary-100 text-primary-700 rounded text-sm"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Recent Scores */}
          {scores.length > 0 && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Recent Scores</h2>
              <div className="space-y-3">
                {scores.slice(0, 3).map((score, idx) => (
                  <div key={idx} className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">
                      {new Date(score.scored_at).toLocaleDateString()}
                    </span>
                    <span className={`font-bold ${
                      score.result.overall_score >= 75 ? 'text-green-600' :
                      score.result.overall_score >= 50 ? 'text-yellow-600' :
                      'text-red-600'
                    }`}>
                      {score.result.overall_score.toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

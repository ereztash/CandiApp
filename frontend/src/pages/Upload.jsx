import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { resumesAPI } from '../services/api'
import toast from 'react-hot-toast'
import {
  CloudArrowUpIcon,
  DocumentTextIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'

export default function Upload() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const navigate = useNavigate()

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0])
      setResult(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false,
  })

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    try {
      const response = await resumesAPI.parse(file)
      setResult(response.data)
      toast.success('Resume parsed successfully!')
    } catch (error) {
      const message = error.response?.data?.message || 'Failed to parse resume'
      toast.error(message)
    } finally {
      setUploading(false)
    }
  }

  const clearFile = () => {
    setFile(null)
    setResult(null)
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-8">Upload Resume</h1>

      {/* Dropzone */}
      {!result && (
        <div className="card mb-6">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-primary-400'
            }`}
          >
            <input {...getInputProps()} />
            <CloudArrowUpIcon className="w-12 h-12 mx-auto text-gray-400 mb-4" />
            {isDragActive ? (
              <p className="text-primary-600">Drop the file here...</p>
            ) : (
              <>
                <p className="text-gray-600 mb-2">
                  Drag & drop a resume here, or click to select
                </p>
                <p className="text-sm text-gray-400">
                  Supported formats: PDF, DOCX, DOC, TXT (max 10MB)
                </p>
              </>
            )}
          </div>

          {/* Selected file */}
          {file && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg flex items-center justify-between">
              <div className="flex items-center">
                <DocumentTextIcon className="w-8 h-8 text-primary-600" />
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-900">{file.name}</p>
                  <p className="text-xs text-gray-500">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
              <button
                onClick={clearFile}
                className="p-1 hover:bg-gray-200 rounded"
              >
                <XMarkIcon className="w-5 h-5 text-gray-500" />
              </button>
            </div>
          )}

          {/* Upload button */}
          {file && (
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="mt-4 w-full btn-primary py-3 disabled:opacity-50"
            >
              {uploading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Parsing...
                </span>
              ) : (
                'Parse Resume'
              )}
            </button>
          )}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-gray-900">Parse Result</h2>
            <div className="flex gap-2">
              <button
                onClick={clearFile}
                className="btn-secondary"
              >
                Upload Another
              </button>
              <button
                onClick={() => navigate(`/resumes/${result.id}`)}
                className="btn-primary"
              >
                View Details
              </button>
            </div>
          </div>

          {/* Parsed data summary */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500">Name</p>
              <p className="font-medium">
                {result.parsed_data?.personal?.full_name || 'Not detected'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Email</p>
              <p className="font-medium">
                {result.parsed_data?.contact?.email || 'Not detected'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Phone</p>
              <p className="font-medium">
                {result.parsed_data?.contact?.phone || 'Not detected'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Experience</p>
              <p className="font-medium">
                {result.parsed_data?.total_experience_years
                  ? `${result.parsed_data.total_experience_years} years`
                  : 'Not detected'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Parsing Time</p>
              <p className="font-medium">
                {result.parsing_time ? `${(result.parsing_time * 1000).toFixed(0)}ms` : 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Skills Found</p>
              <p className="font-medium">
                {result.parsed_data?.technical_skills?.length || 0} technical skills
              </p>
            </div>
          </div>

          {/* Skills */}
          {result.parsed_data?.technical_skills?.length > 0 && (
            <div className="mt-6">
              <p className="text-sm text-gray-500 mb-2">Technical Skills</p>
              <div className="flex flex-wrap gap-2">
                {result.parsed_data.technical_skills.slice(0, 10).map((skill, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm"
                  >
                    {skill}
                  </span>
                ))}
                {result.parsed_data.technical_skills.length > 10 && (
                  <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-sm">
                    +{result.parsed_data.technical_skills.length - 10} more
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

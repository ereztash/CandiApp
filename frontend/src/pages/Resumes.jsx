import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { resumesAPI } from '../services/api'
import toast from 'react-hot-toast'
import {
  DocumentTextIcon,
  TrashIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline'

export default function Resumes() {
  const [resumes, setResumes] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')

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

  const handleDelete = async (id, e) => {
    e.preventDefault()
    e.stopPropagation()

    if (!confirm('Are you sure you want to delete this resume?')) return

    try {
      await resumesAPI.delete(id)
      setResumes(resumes.filter(r => r.id !== id))
      toast.success('Resume deleted')
    } catch (error) {
      toast.error('Failed to delete resume')
    }
  }

  const filteredResumes = resumes.filter(resume => {
    const name = resume.parsed_data?.personal?.full_name || resume.file_name
    return name.toLowerCase().includes(search.toLowerCase())
  })

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Resumes</h1>
        <Link to="/upload" className="btn-primary">
          Upload New
        </Link>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          placeholder="Search resumes..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="input-field pl-10"
        />
      </div>

      {/* Resume list */}
      {filteredResumes.length === 0 ? (
        <div className="card text-center py-12">
          <DocumentTextIcon className="w-12 h-12 mx-auto text-gray-300 mb-4" />
          <p className="text-gray-500">
            {search ? 'No resumes match your search' : 'No resumes yet'}
          </p>
          {!search && (
            <Link to="/upload" className="text-primary-600 hover:text-primary-700 mt-2 inline-block">
              Upload your first resume
            </Link>
          )}
        </div>
      ) : (
        <div className="grid gap-4">
          {filteredResumes.map((resume) => (
            <Link
              key={resume.id}
              to={`/resumes/${resume.id}`}
              className="card hover:shadow-lg transition-shadow"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="p-2 bg-primary-100 rounded-lg">
                    <DocumentTextIcon className="w-6 h-6 text-primary-600" />
                  </div>
                  <div className="ml-4">
                    <p className="font-medium text-gray-900">
                      {resume.parsed_data?.personal?.full_name || 'Unknown Name'}
                    </p>
                    <p className="text-sm text-gray-500">
                      {resume.file_name} • {(resume.file_size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <p className="text-sm text-gray-500">
                      {resume.parsed_data?.contact?.email || 'No email'}
                    </p>
                    <p className="text-xs text-gray-400">
                      {new Date(resume.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={(e) => handleDelete(resume.id, e)}
                    className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg"
                  >
                    <TrashIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {/* Skills preview */}
              {resume.parsed_data?.technical_skills?.length > 0 && (
                <div className="mt-4 flex flex-wrap gap-1">
                  {resume.parsed_data.technical_skills.slice(0, 5).map((skill, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs"
                    >
                      {skill}
                    </span>
                  ))}
                  {resume.parsed_data.technical_skills.length > 5 && (
                    <span className="px-2 py-0.5 text-gray-400 text-xs">
                      +{resume.parsed_data.technical_skills.length - 5}
                    </span>
                  )}
                </div>
              )}
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}

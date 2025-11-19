import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { resumesAPI, healthAPI } from '../services/api'
import {
  DocumentTextIcon,
  CloudArrowUpIcon,
  ChartBarIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline'

export default function Dashboard() {
  const [stats, setStats] = useState({
    totalResumes: 0,
    recentResumes: [],
    systemHealth: null,
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const [resumesRes, healthRes] = await Promise.all([
        resumesAPI.list(0, 5),
        healthAPI.check(),
      ])

      setStats({
        totalResumes: resumesRes.data.length,
        recentResumes: resumesRes.data,
        systemHealth: healthRes.data,
      })
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const quickActions = [
    {
      name: 'Upload Resume',
      description: 'Parse a new resume',
      href: '/upload',
      icon: CloudArrowUpIcon,
      color: 'bg-blue-500',
    },
    {
      name: 'Score Candidates',
      description: 'Match against job requirements',
      href: '/score',
      icon: ChartBarIcon,
      color: 'bg-green-500',
    },
    {
      name: 'View Resumes',
      description: 'Browse parsed resumes',
      href: '/resumes',
      icon: DocumentTextIcon,
      color: 'bg-purple-500',
    },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-8">Dashboard</h1>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center">
            <div className="p-3 bg-blue-100 rounded-lg">
              <DocumentTextIcon className="w-6 h-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-500">Total Resumes</p>
              <p className="text-2xl font-bold text-gray-900">{stats.totalResumes}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="p-3 bg-green-100 rounded-lg">
              <CheckCircleIcon className="w-6 h-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-500">System Status</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats.systemHealth?.status === 'healthy' ? 'Healthy' : 'Degraded'}
              </p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="p-3 bg-purple-100 rounded-lg">
              <ChartBarIcon className="w-6 h-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm text-gray-500">API Version</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats.systemHealth?.version || '2.0.0'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {quickActions.map((action) => (
          <Link
            key={action.name}
            to={action.href}
            className="card hover:shadow-lg transition-shadow"
          >
            <div className={`inline-flex p-3 rounded-lg ${action.color}`}>
              <action.icon className="w-6 h-6 text-white" />
            </div>
            <h3 className="mt-4 text-lg font-medium text-gray-900">{action.name}</h3>
            <p className="mt-1 text-sm text-gray-500">{action.description}</p>
          </Link>
        ))}
      </div>

      {/* Recent Resumes */}
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Resumes</h2>
      <div className="card">
        {stats.recentResumes.length === 0 ? (
          <p className="text-gray-500 text-center py-8">
            No resumes yet.{' '}
            <Link to="/upload" className="text-primary-600 hover:text-primary-700">
              Upload your first resume
            </Link>
          </p>
        ) : (
          <div className="divide-y">
            {stats.recentResumes.map((resume) => (
              <Link
                key={resume.id}
                to={`/resumes/${resume.id}`}
                className="flex items-center py-3 hover:bg-gray-50 -mx-6 px-6"
              >
                <DocumentTextIcon className="w-5 h-5 text-gray-400" />
                <div className="ml-3 flex-1">
                  <p className="text-sm font-medium text-gray-900">
                    {resume.parsed_data?.personal?.full_name || resume.file_name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {new Date(resume.created_at).toLocaleDateString()}
                  </p>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

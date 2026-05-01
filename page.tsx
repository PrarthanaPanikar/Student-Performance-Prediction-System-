'use client'

import { useState, useEffect } from 'react'
import { AlertCircle, TrendingUp, Users, BookOpen, Clock, Target, CheckCircle, XCircle } from 'lucide-react'

// Types for API responses
interface StudentFeatures {
  prior_gpa: number
  attendance_pct: number
  study_hours_wk: number
  commute_min: number
  quiz_avg: number
  assign_avg: number
  midterm: number
  on_time_submit_pct: number
  lms_logins_wk: number
  forum_posts: number
  gender: string
  school_type: string
  parent_edu: string
}

interface PredictionResponse {
  student_id?: string
  risk_probability: number
  risk_level: string
  prediction: boolean
  confidence: number
  timestamp: string
}

interface ExplanationResponse {
  student_id?: string
  top_factors: Array<{
    factor: string
    description: string
    severity: string
  }>
  recommendations: string[]
  risk_drivers: Record<string, number>
}

export default function Home() {
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null)
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  
  // Form state
  const [formData, setFormData] = useState<StudentFeatures>({
    prior_gpa: 3.0,
    attendance_pct: 85,
    study_hours_wk: 12,
    commute_min: 30,
    quiz_avg: 75,
    assign_avg: 78,
    midterm: 72,
    on_time_submit_pct: 90,
    lms_logins_wk: 4,
    forum_posts: 2,
    gender: 'M',
    school_type: 'Public',
    parent_edu: 'Bachelor'
  })

  // Check API status on mount
  useEffect(() => {
    checkApiStatus()
  }, [])

  const checkApiStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/health')
      if (response.ok) {
        setApiStatus('online')
      } else {
        setApiStatus('offline')
      }
    } catch (error) {
      setApiStatus('offline')
    }
  }

  const handleInputChange = (field: keyof StudentFeatures, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const makePrediction = async () => {
    setIsLoading(true)
    try {
      // Make prediction
      const predictResponse = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (!predictResponse.ok) {
        throw new Error('Prediction failed')
      }

      const predictionData: PredictionResponse = await predictResponse.json()
      setPrediction(predictionData)

      // Get explanation
      const explainResponse = await fetch('http://localhost:8000/explain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (explainResponse.ok) {
        const explanationData: ExplanationResponse = await explainResponse.json()
        setExplanation(explanationData)
      }

    } catch (error) {
      console.error('Error making prediction:', error)
      alert('Error making prediction. Please check if the API server is running on localhost:8000')
    } finally {
      setIsLoading(false)
    }
  }

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Low Risk': return 'text-green-600 bg-green-50 border-green-200'
      case 'Medium Risk': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'High Risk': return 'text-red-600 bg-red-50 border-red-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Low Risk': return <CheckCircle className="w-5 h-5 text-green-600" />
      case 'Medium Risk': return <AlertCircle className="w-5 h-5 text-yellow-600" />
      case 'High Risk': return <XCircle className="w-5 h-5 text-red-600" />
      default: return <AlertCircle className="w-5 h-5 text-gray-600" />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <BookOpen className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Student Performance Predictor</h1>
                <p className="text-sm text-gray-500">AI-powered academic risk assessment</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                apiStatus === 'online' ? 'bg-green-100 text-green-800' : 
                apiStatus === 'offline' ? 'bg-red-100 text-red-800' : 
                'bg-yellow-100 text-yellow-800'
              }`}>
                {apiStatus === 'online' ? 'API Online' : 
                 apiStatus === 'offline' ? 'API Offline' : 
                 'Checking...'}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Input Form */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-6 flex items-center">
              <Users className="w-5 h-5 mr-2 text-blue-600" />
              Student Information
            </h2>
            
            <div className="space-y-4">
              {/* Academic Info */}
              <div className="border-b pb-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3">Academic Background</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Prior GPA
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="4"
                      value={formData.prior_gpa}
                      onChange={(e) => handleInputChange('prior_gpa', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Attendance %
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={formData.attendance_pct}
                      onChange={(e) => handleInputChange('attendance_pct', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Study Hours/Week
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="50"
                      value={formData.study_hours_wk}
                      onChange={(e) => handleInputChange('study_hours_wk', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Commute Time (min)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="180"
                      value={formData.commute_min}
                      onChange={(e) => handleInputChange('commute_min', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>

              {/* Performance */}
              <div className="border-b pb-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3">Performance Metrics</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Quiz Average
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={formData.quiz_avg}
                      onChange={(e) => handleInputChange('quiz_avg', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Assignment Average
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={formData.assign_avg}
                      onChange={(e) => handleInputChange('assign_avg', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Midterm Score
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={formData.midterm}
                      onChange={(e) => handleInputChange('midterm', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      On-time Submission %
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={formData.on_time_submit_pct}
                      onChange={(e) => handleInputChange('on_time_submit_pct', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>

              {/* Engagement */}
              <div className="border-b pb-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3">Engagement</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      LMS Logins/Week
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="50"
                      value={formData.lms_logins_wk}
                      onChange={(e) => handleInputChange('lms_logins_wk', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Forum Posts
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={formData.forum_posts}
                      onChange={(e) => handleInputChange('forum_posts', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>

              {/* Demographics */}
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">Demographics</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Gender
                    </label>
                    <select
                      value={formData.gender}
                      onChange={(e) => handleInputChange('gender', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="M">Male</option>
                      <option value="F">Female</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      School Type
                    </label>
                    <select
                      value={formData.school_type}
                      onChange={(e) => handleInputChange('school_type', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="Public">Public</option>
                      <option value="Private">Private</option>
                      <option value="Charter">Charter</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Parent Education
                    </label>
                    <select
                      value={formData.parent_edu}
                      onChange={(e) => handleInputChange('parent_edu', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="High School">High School</option>
                      <option value="Some College">Some College</option>
                      <option value="Bachelor">Bachelor</option>
                      <option value="Master">Master</option>
                      <option value="PhD">PhD</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <button
                onClick={makePrediction}
                disabled={isLoading || apiStatus !== 'online'}
                className="w-full bg-blue-600 text-white py-3 px-4 rounded-md font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
              >
                {isLoading ? (
                  <>
                    <Clock className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Target className="w-4 h-4 mr-2" />
                    Predict Performance
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results */}
          <div className="space-y-6">
            {/* Prediction Result */}
            {prediction && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <TrendingUp className="w-5 h-5 mr-2 text-blue-600" />
                  Prediction Results
                </h2>
                
                <div className="space-y-4">
                  {/* Risk Level */}
                  <div className={`border rounded-lg p-4 ${getRiskColor(prediction.risk_level)}`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getRiskIcon(prediction.risk_level)}
                        <div>
                          <p className="font-semibold">{prediction.risk_level}</p>
                          <p className="text-sm opacity-75">Risk Assessment</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold">{(prediction.risk_probability * 100).toFixed(1)}%</p>
                        <p className="text-sm opacity-75">Pass Probability</p>
                      </div>
                    </div>
                  </div>

                  {/* Prediction Details */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 rounded-lg p-3">
                      <p className="text-sm text-gray-600">Prediction</p>
                      <p className="font-semibold">
                        {prediction.prediction ? 'Will Pass' : 'At Risk of Failing'}
                      </p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <p className="text-sm text-gray-600">Confidence</p>
                      <p className="font-semibold">{(prediction.confidence * 100).toFixed(1)}%</p>
                    </div>
                  </div>

                  {/* Probability Bar */}
                  <div>
                    <div className="flex justify-between text-sm text-gray-600 mb-2">
                      <span>Fail Risk</span>
                      <span>Pass Probability</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div 
                        className="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-3 rounded-full"
                        style={{ width: `${prediction.risk_probability * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Explanation */}
            {explanation && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <AlertCircle className="w-5 h-5 mr-2 text-blue-600" />
                  Risk Analysis & Recommendations
                </h2>
                
                {/* Risk Factors */}
                {explanation.top_factors.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-sm font-medium text-gray-700 mb-3">Key Risk Factors</h3>
                    <div className="space-y-2">
                      {explanation.top_factors.map((factor, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center space-x-3">
                            <div className={`w-2 h-2 rounded-full ${
                              factor.severity === 'high' ? 'bg-red-500' : 'bg-yellow-500'
                            }`} />
                            <span className="text-sm font-medium">{factor.description}</span>
                          </div>
                          <span className={`text-xs px-2 py-1 rounded ${
                            factor.severity === 'high' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {factor.severity}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {explanation.recommendations.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-3">Recommended Actions</h3>
                    <div className="space-y-2">
                      {explanation.recommendations.map((rec, index) => (
                        <div key={index} className="flex items-start space-x-2 p-3 bg-blue-50 rounded-lg">
                          <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                          <span className="text-sm text-gray-700">{rec}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* API Status Warning */}
            {apiStatus === 'offline' && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center space-x-3">
                  <XCircle className="w-5 h-5 text-red-600" />
                  <div>
                    <p className="font-medium text-red-800">API Server Offline</p>
                    <p className="text-sm text-red-600">
                      Please start the FastAPI server on localhost:8000 to enable predictions.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

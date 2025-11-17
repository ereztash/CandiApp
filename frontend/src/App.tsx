/**
 * Main App Component
 * Application entry point with routing
 */
import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider, createTheme, CssBaseline, Box } from '@mui/material'
import ScreeningPage from './pages/ScreeningPage'
import ResultsPage from './pages/ResultsPage'
import AppBar from './components/AppBar'

// Create MUI theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#9c27b0',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 700,
    },
    h5: {
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
  },
})

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Box sx={{ minHeight: '100vh', backgroundColor: 'grey.50' }}>
          <AppBar />
          <Routes>
            <Route path="/" element={<Navigate to="/screening" replace />} />
            <Route path="/screening" element={<ScreeningPage />} />
            <Route path="/results" element={<ResultsPage />} />
          </Routes>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App

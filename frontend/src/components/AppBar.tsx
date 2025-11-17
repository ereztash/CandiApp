/**
 * AppBar Component
 * Top navigation bar
 */
import React from 'react'
import {
  AppBar as MuiAppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
} from '@mui/material'
import { Search, Assessment } from '@mui/icons-material'
import { useNavigate, useLocation } from 'react-router-dom'

export const AppBar: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()

  const isActive = (path: string) => location.pathname === path

  return (
    <MuiAppBar position="sticky" elevation={1}>
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <Typography
            variant="h6"
            component="div"
            sx={{ flexGrow: 1, fontWeight: 'bold' }}
          >
            Resume Screening System
          </Typography>

          <Box display="flex" gap={1}>
            <Button
              color="inherit"
              startIcon={<Search />}
              onClick={() => navigate('/screening')}
              sx={{
                backgroundColor: isActive('/screening')
                  ? 'rgba(255, 255, 255, 0.15)'
                  : 'transparent',
              }}
            >
              Screen Resume
            </Button>
            <Button
              color="inherit"
              startIcon={<Assessment />}
              onClick={() => navigate('/results')}
              sx={{
                backgroundColor: isActive('/results')
                  ? 'rgba(255, 255, 255, 0.15)'
                  : 'transparent',
              }}
            >
              Results
            </Button>
          </Box>
        </Toolbar>
      </Container>
    </MuiAppBar>
  )
}

export default AppBar

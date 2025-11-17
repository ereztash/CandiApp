# Resume Screening System - Frontend

React + TypeScript frontend for the Resume Screening System.

## Features

- **Resume Upload**: Drag & drop interface for resume upload
- **Real-time Screening**: Screen resumes against job requirements
- **Results Dashboard**: View all screening results with filtering
- **Detailed Analysis**: View process-fit scores, archetype detection, and skill matching
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- React 18
- TypeScript
- Material-UI (MUI)
- React Router
- Axios
- Vite

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env`:
```
VITE_API_URL=http://localhost:8000
```

### 3. Run Development Server

```bash
npm run dev
```

The app will be available at http://localhost:3000

### 4. Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Pages

### Screening Page (`/screening`)
- Select a job
- Enter candidate information
- Upload or paste resume text
- View screening results immediately

### Results Page (`/results`)
- View all screened candidates
- Filter by job and decision
- View detailed screening results
- Export results to CSV

## Components

- `AppBar`: Top navigation bar
- `ScreeningResult`: Detailed screening result display
- `ScreeningPage`: Main screening interface
- `ResultsPage`: Results table and filtering

## API Integration

The frontend communicates with the backend API through the `apiService`:

```typescript
import apiService from './services/api'

// Screen a candidate
const response = await apiService.screenCandidate({
  candidate_name: 'John Doe',
  candidate_email: 'john@example.com',
  resume_text: '...',
  job_id: 'uuid'
})

// Get all results
const results = await apiService.listCandidates()
```

## Customization

### Theme

Edit `src/App.tsx` to customize the Material-UI theme:

```typescript
const theme = createTheme({
  palette: {
    primary: { main: '#1976d2' },
    secondary: { main: '#9c27b0' },
  },
})
```

### Archetype Colors

Edit `src/components/ScreeningResult.tsx`:

```typescript
const ARCHETYPE_COLORS: Record<ProcessArchetype, string> = {
  'Innovator': '#9c27b0',
  'Leader': '#1976d2',
  // ...
}
```

## Development

```bash
# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## License

Proprietary - All rights reserved

"""
Enhanced Resume Parser

Improved parsing for:
- Experience sections (dates, companies, titles, descriptions)
- Education sections (degrees, institutions, dates, GPAs)
- Better date extraction and normalization
- Company name disambiguation
- Title standardization
"""

import re
import logging
from typing import List, Optional, Tuple, Dict
from datetime import datetime
from dateutil import parser as date_parser

from .models import Experience, Education, EducationLevel, ParsedData
from .nlp_advanced import AdvancedNLPProcessor, EntityRecognizer

logger = logging.getLogger(__name__)


class EnhancedExperienceParser:
    """
    Enhanced parser for work experience sections.

    Improvements:
    - Better date extraction
    - Company name recognition
    - Title extraction
    - Responsibility and achievement parsing
    """

    def __init__(self):
        """Initialize enhanced experience parser."""
        self.nlp = AdvancedNLPProcessor()
        self.entity_recognizer = EntityRecognizer()

        # Common date formats
        self.date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # 01/2020, 01-2020
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})',  # Jan 2020
            r'(\d{4})',  # 2020
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})',  # Jan 1, 2020
        ]

        # Title patterns
        self.title_indicators = [
            r'(?:title|position|role):\s*(.+?)(?:\n|$)',
            r'(?:as|worked as)\s+(?:a\s+)?([A-Z][A-Za-z\s]+(?:Engineer|Developer|Manager|Director|Analyst))',
        ]

    def parse_experience_block(self, text: str) -> Optional[Experience]:
        """
        Parse a single experience block.

        Args:
            text: Text block for one experience

        Returns:
            Experience object or None
        """
        # Extract company
        company = self._extract_company(text)

        # Extract title
        title = self._extract_title(text)

        # Extract dates
        start_date, end_date, is_current = self._extract_dates(text)

        # Extract location
        location = self._extract_location(text)

        # Extract description, responsibilities, achievements
        description, responsibilities, achievements = self._extract_details(text)

        if not company and not title:
            return None

        return Experience(
            company=company or "Unknown Company",
            title=title or "Unknown Title",
            start_date=start_date,
            end_date=end_date,
            current=is_current,
            location=location,
            description=description,
            responsibilities=responsibilities,
            achievements=achievements,
        )

    def _extract_company(self, text: str) -> Optional[str]:
        """Extract company name from experience text."""
        # Use NER
        companies = self.nlp.extract_companies(text)

        if companies:
            # Return first (most likely to be correct)
            return companies[0]

        # Fallback: look for company name after common keywords
        company_patterns = [
            r'(?:at|@|company:)\s+([A-Z][A-Za-z0-9\s&\.]+)',
            r'([A-Z][A-Za-z0-9\s&\.]+)\s*[,\-]',  # Before comma or dash
        ]

        for pattern in company_patterns:
            match = re.search(pattern, text)
            if match:
                company = match.group(1).strip()
                # Filter out too short or too long
                if 2 < len(company) < 50:
                    return company

        return None

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract job title from experience text."""
        # Try NER
        title = self.entity_recognizer.recognize_job_title(text)
        if title:
            return title

        # Try patterns
        for pattern in self.title_indicators:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Heuristic: First line is often the title
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if it looks like a title (not too long, contains job keywords)
            job_keywords = ["engineer", "developer", "manager", "analyst", "director",
                          "designer", "architect", "scientist", "consultant", "specialist"]

            if any(kw in first_line.lower() for kw in job_keywords) and len(first_line) < 100:
                return first_line

        return None

    def _extract_dates(self, text: str) -> Tuple[Optional[datetime], Optional[datetime], bool]:
        """
        Extract start and end dates from experience text.

        Returns:
            Tuple of (start_date, end_date, is_current)
        """
        # Check for "present" or "current"
        is_current = bool(re.search(r'\b(present|current|now)\b', text.lower()))

        # Extract all dates
        date_strings = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            date_strings.extend(matches)

        # Parse dates
        parsed_dates = []
        for date_str in date_strings:
            try:
                parsed = date_parser.parse(date_str, fuzzy=True)
                parsed_dates.append(parsed)
            except (ValueError, TypeError):
                continue

        # Sort dates
        parsed_dates.sort()

        # Determine start and end
        start_date = None
        end_date = None

        if parsed_dates:
            start_date = parsed_dates[0]

            if len(parsed_dates) > 1:
                end_date = parsed_dates[-1]

        # If current, end_date is None
        if is_current:
            end_date = None

        return start_date, end_date, is_current

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from experience text."""
        locations = self.nlp.extract_locations(text)

        if locations:
            return locations[0]

        return None

    def _extract_details(self, text: str) -> Tuple[Optional[str], List[str], List[str]]:
        """
        Extract description, responsibilities, and achievements.

        Returns:
            Tuple of (description, responsibilities, achievements)
        """
        description = None
        responsibilities = []
        achievements = []

        # Split into lines
        lines = text.split('\n')

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line_lower = line.lower()

            # Detect sections
            if any(kw in line_lower for kw in ["responsibilities", "duties", "role"]):
                current_section = "responsibilities"
                continue
            elif any(kw in line_lower for kw in ["achievements", "accomplishments", "highlights"]):
                current_section = "achievements"
                continue
            elif any(kw in line_lower for kw in ["description", "summary"]):
                current_section = "description"
                continue

            # Add to appropriate section
            # Bullet points (lines starting with -, *, •)
            if re.match(r'^[-*•]\s', line):
                bullet_text = re.sub(r'^[-*•]\s+', '', line)

                if current_section == "achievements":
                    achievements.append(bullet_text)
                else:
                    responsibilities.append(bullet_text)

            # Numbered lists
            elif re.match(r'^\d+[\.)]\s', line):
                numbered_text = re.sub(r'^\d+[\.)]\s+', '', line)

                if current_section == "achievements":
                    achievements.append(numbered_text)
                else:
                    responsibilities.append(numbered_text)

            # Regular text
            else:
                if current_section == "description":
                    description = line if not description else description + " " + line

        # If no description found, use first few lines
        if not description and len(lines) > 1:
            # Skip first line (likely title) and company line
            for line in lines[1:4]:
                if line.strip() and not re.match(r'^[-*•\d]', line):
                    description = line.strip()
                    break

        return description, responsibilities, achievements


class EnhancedEducationParser:
    """
    Enhanced parser for education sections.

    Improvements:
    - Better institution recognition
    - Degree extraction and classification
    - GPA extraction
    - Date parsing
    - Field of study extraction
    """

    def __init__(self):
        """Initialize enhanced education parser."""
        self.nlp = AdvancedNLPProcessor()

        # Degree patterns
        self.degree_patterns = [
            (r'\bPh\.?D\.?\b', EducationLevel.DOCTORATE),
            (r'\bDoctorate\b', EducationLevel.DOCTORATE),
            (r'\bD\.Sc\.?\b', EducationLevel.DOCTORATE),
            (r'\bM\.?S\.?c\.?\b', EducationLevel.MASTER),
            (r'\bM\.?A\.?\b', EducationLevel.MASTER),
            (r'\bMBA\b', EducationLevel.MASTER),
            (r'\bMaster', EducationLevel.MASTER),
            (r'\bB\.?S\.?c\.?\b', EducationLevel.BACHELOR),
            (r'\bB\.?A\.?\b', EducationLevel.BACHELOR),
            (r'\bBachelor', EducationLevel.BACHELOR),
        ]

        # Field of study keywords
        self.field_keywords = [
            "computer science", "engineering", "mathematics", "physics",
            "chemistry", "biology", "business", "economics", "psychology",
            "electrical engineering", "mechanical engineering", "software engineering"
        ]

    def parse_education_block(self, text: str) -> Optional[Education]:
        """
        Parse a single education block.

        Args:
            text: Text block for one education entry

        Returns:
            Education object or None
        """
        # Extract institution
        institution = self._extract_institution(text)

        # Extract degree
        degree, level = self._extract_degree(text)

        # Extract field of study
        field = self._extract_field_of_study(text)

        # Extract dates
        start_date, end_date = self._extract_dates(text)

        # Extract GPA
        gpa = self._extract_gpa(text)

        if not institution:
            return None

        return Education(
            institution=institution,
            degree=degree,
            field_of_study=field,
            start_date=start_date,
            end_date=end_date,
            gpa=gpa,
            level=level,
        )

    def _extract_institution(self, text: str) -> Optional[str]:
        """Extract educational institution name."""
        # Use NER to find organizations
        entities = self.nlp.extract_entities(text)
        orgs = entities.get("ORG", [])

        # Filter for universities/colleges
        edu_keywords = ["university", "college", "institute", "school", "academy"]

        for org in orgs:
            if any(kw in org.lower() for kw in edu_keywords):
                return org

        # Fallback: look for university/college in text
        institution_pattern = r'([A-Z][A-Za-z\s]+(?:University|College|Institute|School))'
        match = re.search(institution_pattern, text)
        if match:
            return match.group(1).strip()

        return None

    def _extract_degree(self, text: str) -> Tuple[Optional[str], Optional[EducationLevel]]:
        """
        Extract degree and its level.

        Returns:
            Tuple of (degree_string, education_level)
        """
        # Try each pattern
        for pattern, level in self.degree_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Try to find full degree text
                degree_match = re.search(
                    rf'{pattern}\s+(?:in\s+)?([A-Za-z\s]+)',
                    text,
                    re.IGNORECASE
                )
                if degree_match:
                    full_degree = match.group(0) + " " + degree_match.group(1)
                    return full_degree.strip(), level
                else:
                    return match.group(0), level

        return None, None

    def _extract_field_of_study(self, text: str) -> Optional[str]:
        """Extract field of study."""
        text_lower = text.lower()

        # Check for "in [FIELD]" or "of [FIELD]" patterns
        field_pattern = r'(?:in|of|major:?)\s+([A-Za-z\s]+)'
        match = re.search(field_pattern, text, re.IGNORECASE)

        if match:
            field = match.group(1).strip()
            # Limit length
            if len(field) < 50:
                return field

        # Check for known fields
        for field in self.field_keywords:
            if field in text_lower:
                return field.title()

        return None

    def _extract_dates(self, text: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract graduation/attendance dates."""
        # Extract all dates
        dates = self.nlp.extract_dates(text)

        if not dates:
            return None, None

        # Sort dates
        dates.sort()

        # Assume first is start, last is graduation
        start_date = dates[0] if dates else None
        end_date = dates[-1] if len(dates) > 1 else dates[0] if dates else None

        return start_date, end_date

    def _extract_gpa(self, text: str) -> Optional[float]:
        """Extract GPA from text."""
        # GPA patterns
        gpa_patterns = [
            r'GPA:\s*([\d.]+)',
            r'(?:GPA|Grade)\s*:\s*([\d.]+)\s*/\s*(\d+)',  # 3.8/4.0
            r'([\d.]+)\s*/\s*4\.0',  # 3.8/4.0
        ]

        for pattern in gpa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    gpa = float(match.group(1))

                    # Normalize to 4.0 scale
                    if gpa > 4.0 and gpa <= 5.0:
                        # Convert from 5.0 scale
                        gpa = gpa * 4.0 / 5.0
                    elif gpa > 5.0 and gpa <= 100:
                        # Convert from 100 scale
                        gpa = gpa * 4.0 / 100

                    if 0 <= gpa <= 4.0:
                        return round(gpa, 2)
                except ValueError:
                    continue

        return None


def parse_resume_enhanced(text: str) -> ParsedData:
    """
    Parse resume with enhanced parsers.

    Args:
        text: Resume text

    Returns:
        ParsedData with improved parsing
    """
    parsed_data = ParsedData()

    # Split into sections
    sections = _split_resume_into_sections(text)

    # Parse experience
    if "experience" in sections:
        exp_parser = EnhancedExperienceParser()
        exp_blocks = _split_into_blocks(sections["experience"])

        for block in exp_blocks:
            exp = exp_parser.parse_experience_block(block)
            if exp:
                parsed_data.experiences.append(exp)

    # Parse education
    if "education" in sections:
        edu_parser = EnhancedEducationParser()
        edu_blocks = _split_into_blocks(sections["education"])

        for block in edu_blocks:
            edu = edu_parser.parse_education_block(block)
            if edu:
                parsed_data.education.append(edu)

    logger.info(f"Enhanced parsing: {len(parsed_data.experiences)} experiences, {len(parsed_data.education)} education")

    return parsed_data


def _split_resume_into_sections(text: str) -> Dict[str, str]:
    """Split resume into major sections."""
    sections = {
        "experience": "",
        "education": "",
        "skills": "",
        "summary": "",
    }

    lines = text.split('\n')
    current_section = None

    for line in lines:
        line_lower = line.lower().strip()

        # Detect section headers
        if any(kw in line_lower for kw in ["experience", "employment", "work history"]):
            current_section = "experience"
        elif any(kw in line_lower for kw in ["education", "academic"]):
            current_section = "education"
        elif any(kw in line_lower for kw in ["skills", "competencies", "technologies"]):
            current_section = "skills"
        elif any(kw in line_lower for kw in ["summary", "objective", "profile"]):
            current_section = "summary"

        # Add to current section
        if current_section and line.strip():
            sections[current_section] += line + "\n"

    return sections


def _split_into_blocks(text: str, min_block_size: int = 50) -> List[str]:
    """
    Split section into individual blocks (one per job/education).

    Uses heuristics:
    - Double line breaks
    - Date patterns as separators
    - Significant indentation changes
    """
    blocks = []
    current_block = ""

    lines = text.split('\n')

    for i, line in enumerate(lines):
        current_block += line + "\n"

        # Check for block separator
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        # Double line break
        if not line.strip() and not next_line.strip():
            if len(current_block.strip()) >= min_block_size:
                blocks.append(current_block.strip())
                current_block = ""

        # Date pattern at start of next line (likely new entry)
        elif re.match(r'^\d{4}|\w+\s+\d{4}', next_line.strip()):
            if len(current_block.strip()) >= min_block_size:
                blocks.append(current_block.strip())
                current_block = ""

    # Add last block
    if current_block.strip() and len(current_block.strip()) >= min_block_size:
        blocks.append(current_block.strip())

    return blocks

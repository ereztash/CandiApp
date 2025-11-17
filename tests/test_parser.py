"""
Unit tests for ResumeParser
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from candiapp.parser import ResumeParser
from candiapp.models import Resume, ParsedData, ContactInfo


class TestResumeParser:
    """Test ResumeParser functionality."""

    def test_parser_initialization(self):
        """Test parser can be initialized."""
        parser = ResumeParser()
        assert parser is not None
        assert parser.enable_nlp is False

    def test_parser_with_nlp(self):
        """Test parser initialization with NLP enabled."""
        parser = ResumeParser(enable_nlp=True)
        assert parser.enable_nlp is True

    def test_supported_formats(self):
        """Test supported file formats."""
        assert ".pdf" in ResumeParser.SUPPORTED_FORMATS
        assert ".docx" in ResumeParser.SUPPORTED_FORMATS
        assert ".doc" in ResumeParser.SUPPORTED_FORMATS
        assert ".txt" in ResumeParser.SUPPORTED_FORMATS

    def test_email_pattern(self):
        """Test email regex pattern."""
        import re
        pattern = ResumeParser.EMAIL_PATTERN

        # Valid emails
        assert re.search(pattern, "test@example.com")
        assert re.search(pattern, "user.name+tag@example.co.uk")

        # Invalid emails
        assert not re.search(pattern, "notanemail")
        assert not re.search(pattern, "@example.com")

    def test_phone_pattern(self):
        """Test phone regex pattern."""
        import re
        pattern = ResumeParser.PHONE_PATTERN

        # Valid phones
        assert re.search(pattern, "123-456-7890")
        assert re.search(pattern, "+1-234-567-8900")
        assert re.search(pattern, "(123) 456-7890")

    def test_linkedin_pattern(self):
        """Test LinkedIn URL pattern."""
        import re
        pattern = ResumeParser.LINKEDIN_PATTERN

        # Valid LinkedIn URLs
        assert re.search(pattern, "https://www.linkedin.com/in/username", re.IGNORECASE)
        assert re.search(pattern, "linkedin.com/in/username", re.IGNORECASE)

    def test_github_pattern(self):
        """Test GitHub URL pattern."""
        import re
        pattern = ResumeParser.GITHUB_PATTERN

        # Valid GitHub URLs
        assert re.search(pattern, "https://github.com/username", re.IGNORECASE)
        assert re.search(pattern, "github.com/username", re.IGNORECASE)

    def test_extract_contact_info(self):
        """Test contact info extraction."""
        parser = ResumeParser()
        text = """
        John Doe
        Email: john.doe@example.com
        Phone: +1-555-123-4567
        LinkedIn: https://linkedin.com/in/johndoe
        GitHub: https://github.com/johndoe
        """

        contact = parser._extract_contact_info(text)

        assert contact.email == "john.doe@example.com"
        assert contact.phone is not None
        assert "johndoe" in contact.linkedin.lower()
        assert "johndoe" in contact.github.lower()

    def test_extract_name(self):
        """Test name extraction."""
        parser = ResumeParser()
        text = """
        John Doe
        Email: john.doe@example.com

        Professional Summary
        Experienced software developer...
        """

        name = parser._extract_name(text)
        assert name == "John Doe"

    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        parser = ResumeParser()

        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent_file.pdf")

    def test_parse_unsupported_format(self):
        """Test parsing an unsupported file format."""
        parser = ResumeParser()

        # Create a temporary file with unsupported extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                parser.parse(temp_path)
        finally:
            Path(temp_path).unlink()


class TestParsedData:
    """Test ParsedData model."""

    def test_parsed_data_initialization(self):
        """Test ParsedData can be initialized."""
        data = ParsedData()
        assert data.full_name is None
        assert isinstance(data.contact, ContactInfo)
        assert isinstance(data.experiences, list)
        assert len(data.experiences) == 0

    def test_parsed_data_to_dict(self):
        """Test conversion to dictionary."""
        data = ParsedData()
        data.full_name = "John Doe"
        data.contact.email = "john@example.com"

        result = data.to_dict()

        assert result["personal"]["full_name"] == "John Doe"
        assert result["contact"]["email"] == "john@example.com"


class TestResume:
    """Test Resume model."""

    def test_resume_initialization(self):
        """Test Resume can be initialized."""
        resume = Resume(
            file_path="/path/to/resume.pdf",
            file_type="pdf",
            file_size=12345,
        )

        assert resume.file_path == "/path/to/resume.pdf"
        assert resume.file_type == "pdf"
        assert resume.file_size == 12345
        assert not resume.is_parsed()

    def test_is_parsed(self):
        """Test is_parsed method."""
        resume = Resume(
            file_path="/path/to/resume.pdf",
            file_type="pdf",
            file_size=12345,
        )

        assert not resume.is_parsed()

        resume.parsed_data = ParsedData()
        assert resume.is_parsed()

    def test_get_field_count(self):
        """Test field count calculation."""
        resume = Resume(
            file_path="/path/to/resume.pdf",
            file_type="pdf",
            file_size=12345,
        )

        # No parsed data
        assert resume.get_field_count() == 0

        # With parsed data
        resume.parsed_data = ParsedData()
        resume.parsed_data.full_name = "John Doe"
        resume.parsed_data.contact.email = "john@example.com"

        count = resume.get_field_count()
        assert count > 0

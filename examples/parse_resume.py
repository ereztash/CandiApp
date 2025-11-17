#!/usr/bin/env python3
"""
Example: Basic Resume Parsing

This example demonstrates how to parse a resume and extract structured data.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from candiapp import ResumeParser


def main():
    """Parse a resume and display results."""
    if len(sys.argv) < 2:
        print("Usage: python parse_resume.py <resume_file>")
        print("\nExample:")
        print("  python parse_resume.py sample_resume.pdf")
        sys.exit(1)

    resume_file = sys.argv[1]

    # Initialize parser
    print("Initializing parser...")
    parser = ResumeParser(enable_nlp=False)

    # Parse resume
    print(f"\nParsing resume: {resume_file}")
    resume = parser.parse(resume_file)

    # Display results
    print("\n" + "=" * 60)
    print("PARSING RESULTS")
    print("=" * 60)

    print(f"\nFile: {resume.file_path}")
    print(f"Type: {resume.file_type}")
    print(f"Size: {resume.file_size} bytes")
    print(f"Parsing Time: {resume.parsing_time:.3f} seconds")
    print(f"Fields Extracted: {resume.get_field_count()}")

    if resume.parsing_errors:
        print(f"\nErrors: {len(resume.parsing_errors)}")
        for error in resume.parsing_errors:
            print(f"  - {error}")
        sys.exit(1)

    # Display parsed data
    if resume.parsed_data:
        data = resume.parsed_data

        print("\n--- PERSONAL INFORMATION ---")
        if data.full_name:
            print(f"Name: {data.full_name}")

        print("\n--- CONTACT ---")
        if data.contact.email:
            print(f"Email: {data.contact.email}")
        if data.contact.phone:
            print(f"Phone: {data.contact.phone}")
        if data.contact.linkedin:
            print(f"LinkedIn: {data.contact.linkedin}")
        if data.contact.github:
            print(f"GitHub: {data.contact.github}")

        print("\n--- SUMMARY ---")
        if data.summary:
            print(data.summary)
        else:
            print("(No summary found)")

        print("\n--- EXPERIENCE ---")
        if data.experiences:
            for i, exp in enumerate(data.experiences, 1):
                print(f"\n{i}. {exp.title} at {exp.company}")
                if exp.start_date or exp.end_date:
                    print(f"   Period: {exp.start_date or 'N/A'} - {exp.end_date or 'Present' if exp.current else 'N/A'}")
        else:
            print("(No experience entries found)")

        if data.total_experience_years:
            print(f"\nTotal Experience: {data.total_experience_years} years")
            if data.experience_level:
                print(f"Level: {data.experience_level.value}")

        print("\n--- EDUCATION ---")
        if data.education:
            for i, edu in enumerate(data.education, 1):
                print(f"\n{i}. {edu.degree or 'Degree'} from {edu.institution}")
                if edu.field_of_study:
                    print(f"   Field: {edu.field_of_study}")
                if edu.start_date or edu.end_date:
                    print(f"   Period: {edu.start_date or 'N/A'} - {edu.end_date or 'N/A'}")
        else:
            print("(No education entries found)")

        print("\n--- SKILLS ---")
        if data.skills:
            print(f"Found {len(data.skills)} skills:")
            for skill in data.skills[:10]:  # Show first 10
                print(f"  - {skill.name}" + (f" ({skill.category})" if skill.category else ""))
        elif data.technical_skills:
            print(f"Technical Skills: {', '.join(data.technical_skills[:10])}")
        else:
            print("(No skills found)")

        print("\n--- LANGUAGES ---")
        if data.languages:
            for lang in data.languages:
                print(f"  - {lang.language}" + (f" ({lang.proficiency})" if lang.proficiency else ""))
        else:
            print("(No languages found)")

        print("\n--- CERTIFICATIONS ---")
        if data.certifications:
            for cert in data.certifications:
                print(f"  - {cert.name}" + (f" by {cert.issuer}" if cert.issuer else ""))
        else:
            print("(No certifications found)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate synthetic resume dataset for benchmarking.

Creates resumes in TXT format with known fields to test parser accuracy.
"""

import os
from pathlib import Path
import json
import random

# Sample data pools
NAMES = [
    "John Smith", "Sarah Johnson", "Michael Chen", "Emily Davis",
    "David Rodriguez", "Jessica Martinez", "Daniel Kim", "Ashley Brown",
    "Christopher Lee", "Amanda Wilson", "יוסי כהן", "רחל לוי",
    "דני אברהם", "מיכל גולן", "עומר שפירא"
]

EMAILS = [
    "john.smith@email.com", "sarah.j@gmail.com", "mchen@yahoo.com",
    "emily.davis@outlook.com", "david.r@proton.me", "jessica.m@email.com",
    "dkim@gmail.com", "ashley.b@email.com", "chris.lee@yahoo.com",
    "amanda.w@gmail.com", "yossi.cohen@gmail.com", "rachel.levy@walla.co.il"
]

PHONES = [
    "+1-555-123-4567", "(555) 234-5678", "555-345-6789",
    "+972-50-123-4567", "050-234-5678", "+1 (555) 456-7890"
]

COMPANIES = [
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix",
    "Salesforce", "Oracle", "IBM", "Intel", "NVIDIA", "Adobe",
    "Wix", "Monday.com", "Check Point", "CyberArk", "Mobileye"
]

TITLES = [
    "Software Engineer", "Senior Software Engineer", "Lead Developer",
    "Backend Developer", "Frontend Developer", "Full Stack Developer",
    "DevOps Engineer", "Data Scientist", "Product Manager",
    "Engineering Manager", "Tech Lead", "Principal Engineer"
]

SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "React", "Angular",
    "Node.js", "Django", "Flask", "Spring Boot", "Docker", "Kubernetes",
    "AWS", "Azure", "GCP", "PostgreSQL", "MongoDB", "Redis",
    "Git", "CI/CD", "Agile", "Scrum", "REST API", "GraphQL",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch"
]

UNIVERSITIES = [
    "Stanford University", "MIT", "Carnegie Mellon University",
    "UC Berkeley", "Harvard University", "Tel Aviv University",
    "Technion", "Hebrew University", "Ben-Gurion University"
]

DEGREES = [
    "B.Sc. in Computer Science",
    "M.Sc. in Computer Science",
    "B.A. in Software Engineering",
    "M.Eng. in Electrical Engineering",
    "Ph.D. in Machine Learning"
]


def generate_resume(resume_id: int) -> tuple[str, dict]:
    """Generate a single resume with known ground truth."""

    # Ground truth data
    ground_truth = {
        "id": resume_id,
        "name": random.choice(NAMES),
        "email": random.choice(EMAILS),
        "phone": random.choice(PHONES),
        "skills": random.sample(SKILLS, k=random.randint(5, 12)),
        "years_experience": random.randint(1, 15),
        "num_experiences": random.randint(2, 5),
        "num_education": random.randint(1, 2),
    }

    # Generate resume text
    resume_text = f"""{ground_truth['name']}
{ground_truth['email']} | {ground_truth['phone']}
LinkedIn: https://linkedin.com/in/{ground_truth['name'].lower().replace(' ', '-')}

PROFESSIONAL SUMMARY
Experienced software professional with {ground_truth['years_experience']} years of expertise in software development,
cloud architecture, and team leadership. Proven track record of delivering high-quality
solutions and mentoring junior developers.

TECHNICAL SKILLS
{', '.join(ground_truth['skills'])}

PROFESSIONAL EXPERIENCE
"""

    # Add work experiences
    experiences = []
    current_year = 2025
    for i in range(ground_truth['num_experiences']):
        company = random.choice(COMPANIES)
        title = random.choice(TITLES)
        years = random.randint(1, 4)
        end_year = current_year - (i * 2)
        start_year = end_year - years

        experiences.append({
            "company": company,
            "title": title,
            "start": start_year,
            "end": end_year if i > 0 else "Present"
        })

        resume_text += f"""
{title} at {company}
{start_year} - {experiences[-1]['end']}
• Developed and maintained scalable applications using modern technologies
• Collaborated with cross-functional teams to deliver features on time
• Implemented CI/CD pipelines and automated testing frameworks
• Mentored junior developers and conducted code reviews
"""

    ground_truth['experiences'] = experiences

    # Add education
    resume_text += "\nEDUCATION\n"
    education = []
    for i in range(ground_truth['num_education']):
        university = random.choice(UNIVERSITIES)
        degree = random.choice(DEGREES)
        grad_year = 2025 - ground_truth['years_experience'] - (i * 2)

        education.append({
            "university": university,
            "degree": degree,
            "year": grad_year
        })

        resume_text += f"""
{degree}
{university}, {grad_year}
"""

    ground_truth['education'] = education

    # Add certifications
    if random.random() > 0.5:
        resume_text += """
CERTIFICATIONS
• AWS Certified Solutions Architect
• Professional Scrum Master (PSM I)
"""
        ground_truth['certifications'] = ["AWS Certified Solutions Architect", "PSM I"]
    else:
        ground_truth['certifications'] = []

    # Add languages
    languages = ["English"]
    if "Hebrew" in ground_truth['name'] or any(ord(c) > 127 for c in ground_truth['name']):
        languages.append("Hebrew")

    resume_text += f"""
LANGUAGES
{', '.join(languages)}
"""
    ground_truth['languages'] = languages

    return resume_text, ground_truth


def main():
    """Generate dataset of resumes."""
    output_dir = Path("data/synthetic_resumes")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_resumes = 50
    ground_truths = []

    print(f"Generating {num_resumes} synthetic resumes...")

    for i in range(num_resumes):
        resume_text, ground_truth = generate_resume(i)

        # Save resume as text file
        resume_file = output_dir / f"resume_{i:03d}.txt"
        resume_file.write_text(resume_text, encoding='utf-8')

        ground_truths.append(ground_truth)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_resumes} resumes")

    # Save ground truth
    ground_truth_file = output_dir / "ground_truth.json"
    ground_truth_file.write_text(
        json.dumps(ground_truths, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    print(f"\n✓ Generated {num_resumes} resumes in {output_dir}")
    print(f"✓ Ground truth saved to {ground_truth_file}")

    # Print statistics
    total_exp = sum(gt['years_experience'] for gt in ground_truths)
    avg_exp = total_exp / num_resumes
    total_skills = sum(len(gt['skills']) for gt in ground_truths)
    avg_skills = total_skills / num_resumes

    print(f"\nDataset Statistics:")
    print(f"  Total resumes: {num_resumes}")
    print(f"  Average experience: {avg_exp:.1f} years")
    print(f"  Average skills per resume: {avg_skills:.1f}")
    print(f"  Total experiences: {sum(gt['num_experiences'] for gt in ground_truths)}")
    print(f"  Total education entries: {sum(gt['num_education'] for gt in ground_truths)}")


if __name__ == "__main__":
    main()

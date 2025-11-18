"""
Synthetic Resume Data Generator for Benchmarking

Generates realistic synthetic resume data for testing and benchmarking.
"""

import sys
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from candiapp.models import (
    ParsedData,
    ContactInfo,
    Experience,
    Education,
    Skill,
    Language,
    Certification,
    Project,
    ExperienceLevel,
    EducationLevel,
)


class SyntheticResumeGenerator:
    """
    Generates synthetic but realistic resume data.
    """

    FIRST_NAMES = [
        "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa",
        "Daniel", "Jennifer", "James", "Mary", "William", "Patricia", "Richard",
        "Yossi", "Rachel", "Aviv", "Noa", "Eli", "Tamar", "Oren", "Maya",
    ]

    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Cohen", "Levi", "Goldberg", "Katz",
        "Friedman", "Shapiro", "Ben-David", "Sharon", "Avraham",
    ]

    COMPANIES = [
        "Google", "Microsoft", "Amazon", "Apple", "Meta", "Intel", "NVIDIA",
        "IBM", "Oracle", "Salesforce", "Adobe", "Cisco", "Dell", "HP",
        "Startup Inc", "Tech Corp", "Innovation Labs", "Digital Solutions",
        "CloudTech", "DataWorks", "AI Systems", "DevHub", "CodeFactory",
    ]

    JOB_TITLES = {
        ExperienceLevel.ENTRY: [
            "Junior Developer", "Associate Engineer", "Intern", "Trainee Developer",
            "Entry-level Analyst", "Junior Programmer",
        ],
        ExperienceLevel.JUNIOR: [
            "Software Developer", "Software Engineer", "Developer", "Programmer",
            "Data Analyst", "QA Engineer",
        ],
        ExperienceLevel.MID: [
            "Software Engineer", "Backend Developer", "Frontend Developer",
            "Full Stack Developer", "DevOps Engineer", "Data Scientist",
        ],
        ExperienceLevel.SENIOR: [
            "Senior Software Engineer", "Senior Developer", "Senior Data Scientist",
            "Principal Engineer", "Staff Engineer", "Lead Developer",
        ],
        ExperienceLevel.LEAD: [
            "Tech Lead", "Team Lead", "Engineering Lead", "Lead Architect",
            "Senior Staff Engineer", "Distinguished Engineer",
        ],
        ExperienceLevel.EXECUTIVE: [
            "Engineering Manager", "Director of Engineering", "VP Engineering",
            "CTO", "Chief Architect", "Head of Development",
        ],
    }

    UNIVERSITIES = [
        "MIT", "Stanford University", "Harvard University", "UC Berkeley",
        "Carnegie Mellon", "University of Washington", "Columbia University",
        "Technion", "Tel Aviv University", "Hebrew University", "Ben-Gurion University",
        "Weizmann Institute", "IDC Herzliya",
    ]

    DEGREES = {
        EducationLevel.BACHELOR: ["B.Sc.", "B.A.", "Bachelor of Science", "Bachelor of Arts"],
        EducationLevel.MASTER: ["M.Sc.", "M.A.", "Master of Science", "MBA"],
        EducationLevel.DOCTORATE: ["Ph.D.", "Doctorate", "D.Sc."],
    }

    FIELDS_OF_STUDY = [
        "Computer Science", "Software Engineering", "Electrical Engineering",
        "Data Science", "Information Systems", "Mathematics", "Physics",
        "Computer Engineering", "Artificial Intelligence", "Business Administration",
    ]

    PROGRAMMING_SKILLS = [
        "Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "Ruby",
        "PHP", "Swift", "Kotlin", "TypeScript", "Scala", "R", "MATLAB",
    ]

    WEB_SKILLS = [
        "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring",
        "Express", "FastAPI", "ASP.NET", "Ruby on Rails", "Laravel",
    ]

    DATA_SKILLS = [
        "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
        "Pandas", "NumPy", "Spark", "Hadoop", "Kafka", "Airflow",
    ]

    ML_SKILLS = [
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
        "scikit-learn", "Keras", "NLP", "Computer Vision", "MLOps",
    ]

    CLOUD_SKILLS = [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
        "Jenkins", "GitLab CI", "GitHub Actions", "CircleCI",
    ]

    SOFT_SKILLS = [
        "Leadership", "Communication", "Problem Solving", "Team Player",
        "Critical Thinking", "Creativity", "Time Management", "Adaptability",
        "Collaboration", "Mentoring", "Public Speaking", "Project Management",
    ]

    CERTIFICATIONS = [
        "AWS Certified Solutions Architect", "Google Cloud Professional",
        "Azure Administrator", "PMP", "Scrum Master", "CISSP",
        "CKA (Kubernetes)", "TensorFlow Developer", "Oracle Certified",
    ]

    def generate_resume(
        self,
        experience_level: ExperienceLevel = None,
        seed: int = None
    ) -> ParsedData:
        """
        Generate a single synthetic resume.

        Args:
            experience_level: Desired experience level (random if None)
            seed: Random seed for reproducibility

        Returns:
            ParsedData object with synthetic data
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Random experience level if not specified
        if experience_level is None:
            experience_level = random.choice(list(ExperienceLevel))

        # Generate personal info
        first_name = random.choice(self.FIRST_NAMES)
        last_name = random.choice(self.LAST_NAMES)
        full_name = f"{first_name} {last_name}"

        # Generate contact info
        contact = ContactInfo(
            email=f"{first_name.lower()}.{last_name.lower()}@example.com",
            phone=f"+{random.randint(1, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            linkedin=f"https://linkedin.com/in/{first_name.lower()}{last_name.lower()}" if random.random() > 0.3 else None,
            github=f"https://github.com/{first_name.lower()}{last_name.lower()}" if random.random() > 0.5 else None,
        )

        # Generate experience based on level
        experiences = self._generate_experiences(experience_level)
        total_years = self._calculate_total_years(experiences)

        # Generate education
        education = self._generate_education(experience_level)

        # Generate skills
        skills = self._generate_skills(experience_level)
        technical_skills = self._generate_technical_skills(experience_level)
        soft_skills = random.sample(self.SOFT_SKILLS, k=random.randint(3, 6))

        # Generate optional sections
        languages = self._generate_languages()
        certifications = self._generate_certifications(experience_level)
        projects = self._generate_projects(experience_level)

        # Generate summary
        summary = self._generate_summary(full_name, experience_level, total_years)

        return ParsedData(
            full_name=full_name,
            first_name=first_name,
            last_name=last_name,
            contact=contact,
            summary=summary,
            experiences=experiences,
            total_experience_years=total_years,
            experience_level=experience_level,
            education=education,
            skills=skills,
            technical_skills=technical_skills,
            soft_skills=soft_skills,
            languages=languages,
            certifications=certifications,
            projects=projects,
        )

    def _generate_experiences(self, level: ExperienceLevel) -> List[Experience]:
        """Generate work experiences based on level."""
        # Number of jobs based on experience level
        num_jobs = {
            ExperienceLevel.ENTRY: random.randint(0, 2),
            ExperienceLevel.JUNIOR: random.randint(1, 3),
            ExperienceLevel.MID: random.randint(2, 4),
            ExperienceLevel.SENIOR: random.randint(3, 5),
            ExperienceLevel.LEAD: random.randint(4, 6),
            ExperienceLevel.EXECUTIVE: random.randint(5, 8),
        }

        experiences = []
        current_date = datetime.now()

        # Start from current and go backwards
        for i in range(num_jobs[level]):
            # Determine seniority for this job
            if i == 0:
                job_level = level
            else:
                # Earlier jobs were likely at lower levels
                levels = list(ExperienceLevel)
                current_idx = levels.index(level)
                earlier_idx = max(0, current_idx - i)
                job_level = levels[earlier_idx]

            # Job duration (in months)
            duration_months = random.randint(12, 48)

            # Calculate dates
            if i == 0:
                # Current or most recent job
                current_job = random.random() > 0.3
                end_date = None if current_job else current_date - timedelta(days=random.randint(0, 365))
                start_date = (end_date or current_date) - timedelta(days=duration_months * 30)
            else:
                # Previous job ends when next job started
                next_job_start = experiences[-1].start_date
                gap_months = random.randint(0, 6)  # Career gap
                end_date = next_job_start - timedelta(days=gap_months * 30)
                start_date = end_date - timedelta(days=duration_months * 30)
                current_job = False

            # Generate job details
            company = random.choice(self.COMPANIES)
            title = random.choice(self.JOB_TITLES[job_level])

            # Generate responsibilities (more for senior positions)
            num_responsibilities = random.randint(2, 5) + (levels.index(job_level))
            responsibilities = [
                f"Responsibility {j+1} for {title} role"
                for j in range(num_responsibilities)
            ]

            # Generate achievements (fewer than responsibilities)
            num_achievements = random.randint(1, 3)
            achievements = [
                f"Achievement {j+1} in {company}"
                for j in range(num_achievements)
            ]

            experiences.append(Experience(
                company=company,
                title=title,
                start_date=start_date,
                end_date=end_date,
                current=current_job,
                location=random.choice(["Tel Aviv", "New York", "San Francisco", "London", "Remote"]),
                description=f"Working as {title} at {company}",
                responsibilities=responsibilities,
                achievements=achievements,
            ))

        return experiences

    def _generate_education(self, level: ExperienceLevel) -> List[Education]:
        """Generate education entries."""
        education = []

        # Everyone has at least bachelor's
        has_bachelor = True
        has_master = level in [ExperienceLevel.SENIOR, ExperienceLevel.LEAD, ExperienceLevel.EXECUTIVE] and random.random() > 0.4
        has_phd = level == ExperienceLevel.EXECUTIVE and random.random() > 0.7

        current_year = datetime.now().year

        if has_phd:
            # Ph.D.
            graduation_year = current_year - random.randint(5, 15)
            education.append(Education(
                institution=random.choice(self.UNIVERSITIES),
                degree=random.choice(self.DEGREES[EducationLevel.DOCTORATE]),
                field_of_study=random.choice(self.FIELDS_OF_STUDY),
                start_date=datetime(graduation_year - 4, 9, 1),
                end_date=datetime(graduation_year, 6, 1),
                level=EducationLevel.DOCTORATE,
                gpa=round(random.uniform(3.5, 4.0), 2),
            ))

        if has_master or has_phd:
            # Master's
            graduation_year = current_year - random.randint(3, 10) if not has_phd else graduation_year - 4
            education.append(Education(
                institution=random.choice(self.UNIVERSITIES),
                degree=random.choice(self.DEGREES[EducationLevel.MASTER]),
                field_of_study=random.choice(self.FIELDS_OF_STUDY),
                start_date=datetime(graduation_year - 2, 9, 1),
                end_date=datetime(graduation_year, 6, 1),
                level=EducationLevel.MASTER,
                gpa=round(random.uniform(3.3, 4.0), 2),
            ))

        if has_bachelor:
            # Bachelor's
            years_since_bachelor = {
                ExperienceLevel.ENTRY: random.randint(0, 2),
                ExperienceLevel.JUNIOR: random.randint(2, 5),
                ExperienceLevel.MID: random.randint(5, 10),
                ExperienceLevel.SENIOR: random.randint(8, 15),
                ExperienceLevel.LEAD: random.randint(12, 20),
                ExperienceLevel.EXECUTIVE: random.randint(15, 25),
            }

            graduation_year = current_year - years_since_bachelor[level]
            if has_master:
                graduation_year = min(graduation_year, education[0].end_date.year - 2)

            education.append(Education(
                institution=random.choice(self.UNIVERSITIES),
                degree=random.choice(self.DEGREES[EducationLevel.BACHELOR]),
                field_of_study=random.choice(self.FIELDS_OF_STUDY),
                start_date=datetime(graduation_year - 4, 9, 1),
                end_date=datetime(graduation_year, 6, 1),
                level=EducationLevel.BACHELOR,
                gpa=round(random.uniform(3.0, 4.0), 2),
            ))

        return education

    def _generate_skills(self, level: ExperienceLevel) -> List[Skill]:
        """Generate skill objects with proficiency."""
        skills = []

        # More skills for senior levels
        num_skills = {
            ExperienceLevel.ENTRY: random.randint(3, 6),
            ExperienceLevel.JUNIOR: random.randint(5, 8),
            ExperienceLevel.MID: random.randint(7, 12),
            ExperienceLevel.SENIOR: random.randint(10, 15),
            ExperienceLevel.LEAD: random.randint(12, 18),
            ExperienceLevel.EXECUTIVE: random.randint(15, 20),
        }

        # Sample from different categories
        all_skills = (
            self.PROGRAMMING_SKILLS +
            self.WEB_SKILLS +
            self.DATA_SKILLS +
            self.ML_SKILLS +
            self.CLOUD_SKILLS
        )

        selected_skills = random.sample(all_skills, k=min(num_skills[level], len(all_skills)))

        for skill_name in selected_skills:
            # Determine category
            if skill_name in self.PROGRAMMING_SKILLS:
                category = "Programming"
            elif skill_name in self.WEB_SKILLS:
                category = "Web"
            elif skill_name in self.DATA_SKILLS:
                category = "Data"
            elif skill_name in self.ML_SKILLS:
                category = "ML/AI"
            else:
                category = "Cloud"

            # Proficiency based on level
            proficiency_levels = ["Beginner", "Intermediate", "Advanced", "Expert"]
            level_idx = list(ExperienceLevel).index(level)
            proficiency = proficiency_levels[min(level_idx // 2, 3)]

            # Years of experience
            years = random.randint(1, min(15, level_idx * 2 + 3))

            skills.append(Skill(
                name=skill_name,
                category=category,
                proficiency=proficiency,
                years_of_experience=years,
            ))

        return skills

    def _generate_technical_skills(self, level: ExperienceLevel) -> List[str]:
        """Generate list of technical skills."""
        num_skills = {
            ExperienceLevel.ENTRY: random.randint(5, 10),
            ExperienceLevel.JUNIOR: random.randint(8, 15),
            ExperienceLevel.MID: random.randint(12, 20),
            ExperienceLevel.SENIOR: random.randint(15, 25),
            ExperienceLevel.LEAD: random.randint(20, 30),
            ExperienceLevel.EXECUTIVE: random.randint(25, 35),
        }

        all_skills = (
            self.PROGRAMMING_SKILLS +
            self.WEB_SKILLS +
            self.DATA_SKILLS +
            self.ML_SKILLS +
            self.CLOUD_SKILLS
        )

        return random.sample(all_skills, k=min(num_skills[level], len(all_skills)))

    def _generate_languages(self) -> List[Language]:
        """Generate language proficiencies."""
        languages = []

        # English (almost always)
        if random.random() > 0.1:
            languages.append(Language(
                language="English",
                proficiency=random.choice(["Native", "Fluent", "Professional"])
            ))

        # Hebrew (50% chance)
        if random.random() > 0.5:
            languages.append(Language(
                language="Hebrew",
                proficiency=random.choice(["Native", "Fluent", "Professional"])
            ))

        # Other languages (30% chance)
        if random.random() > 0.7:
            other = random.choice(["Spanish", "French", "German", "Chinese", "Russian"])
            languages.append(Language(
                language=other,
                proficiency=random.choice(["Basic", "Intermediate", "Professional"])
            ))

        return languages

    def _generate_certifications(self, level: ExperienceLevel) -> List[Certification]:
        """Generate certifications."""
        certifications = []

        # More certifications for senior levels
        num_certs = {
            ExperienceLevel.ENTRY: 0,
            ExperienceLevel.JUNIOR: random.randint(0, 1),
            ExperienceLevel.MID: random.randint(1, 2),
            ExperienceLevel.SENIOR: random.randint(1, 3),
            ExperienceLevel.LEAD: random.randint(2, 4),
            ExperienceLevel.EXECUTIVE: random.randint(2, 5),
        }

        if num_certs[level] > 0:
            selected = random.sample(self.CERTIFICATIONS, k=min(num_certs[level], len(self.CERTIFICATIONS)))
            for cert_name in selected:
                issue_year = datetime.now().year - random.randint(0, 5)
                certifications.append(Certification(
                    name=cert_name,
                    issuer=cert_name.split()[0],  # First word as issuer
                    issue_date=datetime(issue_year, random.randint(1, 12), 1),
                ))

        return certifications

    def _generate_projects(self, level: ExperienceLevel) -> List[Project]:
        """Generate personal/professional projects."""
        projects = []

        # More projects for mid to senior levels
        if level in [ExperienceLevel.MID, ExperienceLevel.SENIOR, ExperienceLevel.LEAD]:
            num_projects = random.randint(1, 3)

            for i in range(num_projects):
                projects.append(Project(
                    name=f"Project {i+1}",
                    description=f"Personal project involving advanced technologies",
                    technologies=random.sample(self.PROGRAMMING_SKILLS + self.WEB_SKILLS, k=random.randint(2, 5)),
                    url=f"https://github.com/user/project{i+1}" if random.random() > 0.5 else None,
                ))

        return projects

    def _generate_summary(self, name: str, level: ExperienceLevel, years: float) -> str:
        """Generate professional summary."""
        templates = [
            f"{name} is an experienced professional with {years:.1f} years in software development.",
            f"Seasoned {level.value} engineer with {years:.1f} years of industry experience.",
            f"Passionate developer with {years:.1f} years building innovative solutions.",
            f"Results-driven professional with {years:.1f} years in technology sector.",
        ]

        return random.choice(templates)

    def _calculate_total_years(self, experiences: List[Experience]) -> float:
        """Calculate total years of experience."""
        total_days = 0

        for exp in experiences:
            if exp.start_date:
                end = exp.end_date or datetime.now()
                days = (end - exp.start_date).days
                total_days += days

        return total_days / 365.25

    def generate_dataset(
        self,
        n_samples: int = 100,
        distribution: dict = None
    ) -> Tuple[List[ParsedData], np.ndarray]:
        """
        Generate a dataset of synthetic resumes.

        Args:
            n_samples: Number of resumes to generate
            distribution: Distribution of experience levels (default: uniform)

        Returns:
            Tuple of (resumes, labels) where labels are experience level indices
        """
        if distribution is None:
            # Uniform distribution
            distribution = {level: 1/6 for level in ExperienceLevel}

        resumes = []
        labels = []

        levels = list(ExperienceLevel)

        for i in range(n_samples):
            # Sample level based on distribution
            level = np.random.choice(
                levels,
                p=[distribution[lvl] for lvl in levels]
            )

            resume = self.generate_resume(experience_level=level, seed=i)
            resumes.append(resume)
            labels.append(levels.index(level))

        return resumes, np.array(labels)

"""
generate_dataset.py -- builds the evaluation dataset
====================================================
* 40 synthetic, diverse resumes (75% PDF, 25% TXT) across 8 role archetypes,
  3 seniority levels, varied section headings / ordering / date formats.
* 8 job descriptions with explicit must-have requirements.
* data/ground_truth.json -- true archetype, skills, per-skill years, education
  for every resume + relevance rules for every JD (used by evaluate.py).

All people / companies are fictional.   Usage:  python generate_dataset.py [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from resume_rag import extract_skills, merge_intervals, months_in, today_month_index

NOW = today_month_index()
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ----------------------------- archetype definitions ------------------------ #
# core: always present in every job (so "N+ years of <core>" is well defined)
# extra: sampled per job.  count = number of resumes of this archetype.
ARCH = {
    "ml_engineer": dict(count=6,
        titles={"junior": "Junior ML Engineer", "mid": "Machine Learning Engineer", "senior": "Senior ML Engineer"},
        core=["Python", "Machine Learning", "PyTorch"],
        extra=["TensorFlow", "Docker", "AWS", "MLOps", "Kubernetes", "Deep Learning", "NLP", "Computer Vision",
               "LLMs", "RAG", "scikit-learn", "Spark", "FastAPI", "SQL"],
        edu=[("Computer Science", 0.6), ("Electrical Engineering", 0.2), ("Applied Mathematics", 0.2)],
        flavor=["Trained and deployed models with {a} and {b}, improving offline F1 by {n} points.",
                "Cut inference latency by {n}% by profiling and optimising services built on {a}.",
                "Built feature engineering workflows with {a} and {b} shared across {t} product teams.",
                "Owned the model lifecycle from prototype to production using {a}."]),
    "data_scientist": dict(count=6,
        titles={"junior": "Data Analyst", "mid": "Data Scientist", "senior": "Senior Data Scientist"},
        core=["Python", "SQL", "Statistics"],
        extra=["Machine Learning", "A/B Testing", "scikit-learn", "Pandas", "NumPy", "Tableau", "R"][:6] +
              ["Spark", "Snowflake", "NLP"],
        edu=[("Statistics", 0.4), ("Economics", 0.2), ("Computer Science", 0.2), ("Mathematics", 0.2)],
        flavor=["Delivered analyses with {a} and {b} that drove a {n}% lift in conversion.",
                "Designed and read out {t} controlled studies, partnering with product leaders and using {a}.",
                "Built forecasting and segmentation work in {a}, adopted by finance and marketing.",
                "Created self-serve dashboards with {a} used weekly by {t} stakeholders."]),
    "data_engineer": dict(count=5,
        titles={"junior": "Junior Data Engineer", "mid": "Data Engineer", "senior": "Senior Data Engineer"},
        core=["SQL", "Spark", "Python"],
        extra=["Airflow", "Kafka", "dbt", "Snowflake", "BigQuery", "AWS", "GCP", "ETL", "PostgreSQL", "Docker", "Scala"],
        edu=[("Computer Science", 0.6), ("Information Systems", 0.2), ("Mathematics", 0.2)],
        flavor=["Built batch and streaming ingestion workflows with {a} and {b} processing {n}TB daily.",
                "Migrated legacy warehouse jobs to {a}, cutting runtime by {n}%.",
                "Introduced data quality checks and lineage tracking across {a} and {b}.",
                "Modelled analytics layers in {a} consumed by {t} downstream teams."]),
    "backend_java": dict(count=5,
        titles={"junior": "Junior Java Developer", "mid": "Backend Engineer", "senior": "Senior Backend Engineer"},
        core=["Java", "Spring Boot", "Microservices"],
        extra=["REST APIs", "PostgreSQL", "Kafka", "Docker", "Kubernetes", "AWS", "Redis", "MySQL", "Jenkins", "GraphQL"],
        edu=[("Computer Science", 0.7), ("Software Engineering", 0.3)],
        flavor=["Designed and shipped services with {a} and {b} handling {n}k requests per second.",
                "Decomposed a monolith into independently deployable services built on {a}.",
                "Hardened payment flows and cut p99 latency by {n}% through query and cache tuning with {a}.",
                "Mentored {t} engineers and led design reviews for systems using {a} and {b}."]),
    "frontend": dict(count=5,
        titles={"junior": "Junior Frontend Developer", "mid": "Frontend Engineer", "senior": "Senior Frontend Engineer"},
        core=["React", "JavaScript", "HTML/CSS"],
        extra=["TypeScript", "TypeScript", "GraphQL", "REST APIs", "Node.js", "Figma", "Vue", "Angular", "CI/CD"],
        edu=[("Computer Science", 0.5), ("Graphic Design", 0.2), ("Information Technology", 0.3)],
        flavor=["Built accessible, responsive interfaces with {a} and {b} used by {n}k monthly users.",
                "Improved page load time by {n}% through code splitting and rendering work in {a}.",
                "Created a shared component library in {a} adopted by {t} product squads.",
                "Partnered with designers to ship polished features using {a} and {b}."]),
    "devops": dict(count=5,
        titles={"junior": "Junior DevOps Engineer", "mid": "DevOps Engineer", "senior": "Senior Site Reliability Engineer"},
        core=["Docker", "Linux", "CI/CD", "Kubernetes"],
        extra=["Terraform", "Terraform", "Terraform", "AWS", "AWS", "Prometheus", "Grafana", "Jenkins", "Python", "Azure"],
        edu=[("Computer Science", 0.5), ("Information Technology", 0.3), ("Computer Engineering", 0.2)],
        flavor=["Automated infrastructure provisioning with {a} and {b}, reducing setup time from days to minutes.",
                "Raised service availability to 99.9{n}% by building alerting and runbooks around {a}.",
                "Standardised delivery workflows on {a}, enabling {t} teams to deploy several times a day.",
                "Cut cloud spend by {n}% through capacity planning and rightsizing with {a} and {b}."]),
    "mobile": dict(count=4,
        titles={"junior": "Junior Mobile Developer", "mid": "Mobile Engineer", "senior": "Senior Mobile Engineer"},
        core=[],  # chosen per platform below
        extra=["REST APIs", "GraphQL", "CI/CD", "Agile", "Figma"],
        edu=[("Computer Science", 0.7), ("Software Engineering", 0.3)],
        flavor=["Shipped consumer apps built with {a} and {b} to {n}M downloads.",
                "Reduced crash rate by {n}% through profiling and release hygiene around {a}.",
                "Built offline-first features with {a} for {t} markets.",
                "Led adoption of a modern architecture for apps using {a} and {b}."]),
    "product_manager": dict(count=4,
        titles={"junior": "Associate Product Manager", "mid": "Product Manager", "senior": "Senior Product Manager"},
        core=["Product Management", "Agile"],
        extra=["SQL", "JIRA", "Tableau", "A/B Testing", "Figma", "Excel"],
        edu=[("Business Administration", 0.5), ("Economics", 0.3), ("Computer Science", 0.2)],
        flavor=["Defined strategy and priorities for a platform used by {n}k customers, guided by {a} and {b}.",
                "Ran discovery with {t} customer segments and translated findings into plans tracked in {a}.",
                "Grew activation by {n}% by pairing quantitative analysis in {a} with user research.",
                "Aligned engineering, design and sales around outcomes, reporting through {a}."]),
}
GENERIC_FLAVOR = ["Collaborated across teams to deliver milestones on schedule using {a}.",
                  "Improved reliability and documentation for systems that rely on {a} and {b}.",
                  "Led code reviews and onboarding for {t} new hires working with {a}."]

# surface variations to exercise synonym / alias handling
SURFACE = {
    "Kubernetes": ["Kubernetes", "k8s"], "PostgreSQL": ["PostgreSQL", "Postgres"],
    "Machine Learning": ["machine learning", "ML"], "Deep Learning": ["deep learning", "neural networks"],
    "LLMs": ["LLMs", "large language models"], "NLP": ["NLP", "natural language processing"],
    "Spark": ["Spark", "PySpark"], "scikit-learn": ["scikit-learn", "sklearn"], "JavaScript": ["JavaScript", "JS"],
    "RAG": ["RAG", "retrieval-augmented generation"], "AWS": ["AWS", "Amazon Web Services"],
    "GCP": ["GCP", "Google Cloud"], "CI/CD": ["CI/CD", "continuous integration"], "ETL": ["ETL", "data pipelines"],
    "A/B Testing": ["A/B testing", "A/B tests"], "Statistics": ["statistics", "statistical modeling"],
    "Product Management": ["product management", "product roadmap"], "Node.js": ["Node.js", "NodeJS"],
    "Microservices": ["microservices"], "Spring Boot": ["Spring Boot"], "HTML/CSS": ["HTML5", "CSS3"],
    "REST APIs": ["REST APIs", "RESTful APIs"], "Agile": ["Agile", "Scrum"],
}
CERTS = {"AWS": "AWS Certified Solutions Architect", "Kubernetes": "Certified Kubernetes Administrator (CKA)",
         "Terraform": "HashiCorp Certified: Terraform Associate", "Agile": "Certified Scrum Master (CSM)",
         "Python": "PCEP Certified Python Entry-Level Programmer"}
# "R" is not in the taxonomy; drop it if a template pulled it in
_NOT_IN_TAXONOMY = {"R"}

FIRST = ["Aarav", "Priya", "Wei", "Mei", "Carlos", "Sofia", "Liam", "Olivia", "Noah", "Amara", "Chidi", "Fatima",
         "Hiroshi", "Yuki", "Ivan", "Elena", "Mateo", "Lucía", "Omar", "Layla", "Ethan", "Emma", "Rahul", "Ananya",
         "Jin", "Seo-yeon", "Tomás", "Ingrid", "Kwame", "Zainab", "Daniel", "Hannah", "Arjun", "Divya", "Pavel",
         "Marta", "Samuel", "Grace", "Diego", "Nadia"]
LAST = ["Sharma", "Patel", "Chen", "Wang", "Garcia", "Rossi", "Smith", "Johnson", "Okafor", "Hassan", "Tanaka",
        "Sato", "Petrov", "Novak", "Fernandez", "Ruiz", "Khan", "Haddad", "Miller", "Brown", "Reddy", "Iyer",
        "Kim", "Park", "Silva", "Larsen", "Mensah", "Bello", "Cohen", "Weber", "Nair", "Gupta", "Volkov",
        "Kowalski", "Adams", "Lee", "Torres", "Ali", "Moreau", "Singh"]
COMPANIES = ["Northwind Labs", "Brightpath Systems", "Helio Analytics", "Cobalt Cloud", "Mapleleaf Digital",
             "Quanta Retail", "BlueOrbit", "Fernwood Health", "Lumen Fintech", "Tidewater Logistics", "Pioneer Mobility",
             "Orchid Software", "Vertex Media", "Sable Security", "Ironbridge Bank", "Solstice Energy",
             "Nimbus Games", "Atlas Travel", "Granite Insurance", "Kestrel Robotics"]
UNIS = ["State University", "Institute of Technology", "National University", "City College", "Tech University",
        "Polytechnic University", "Metropolitan University"]
CITIES = ["Bengaluru", "Hyderabad", "Austin, TX", "Seattle, WA", "London", "Toronto", "Berlin", "Singapore",
          "New York, NY", "Pune", "Remote", "San Francisco, CA", "Sydney"]

HEADINGS = {
    "summary": ["SUMMARY", "PROFESSIONAL SUMMARY", "PROFILE"],
    "skills": ["SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES"],
    "experience": ["EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE"],
    "education": ["EDUCATION", "ACADEMIC BACKGROUND"],
    "projects": ["PROJECTS", "SELECTED PROJECTS"],
    "certifications": ["CERTIFICATIONS"],
}
LEVEL_YEARS = {"junior": (1.5, 3.0), "mid": (3.5, 6.0), "senior": (6.5, 12.0)}
LEVEL_CYCLE = ["senior", "mid", "junior", "senior", "mid", "senior", "junior", "mid"]


def surf(rng, skill):
    return rng.choice(SURFACE.get(skill, [skill]))


def idx_to_str(idx, fmt):
    y, m = divmod(idx, 12)
    return f"{MONTH_ABBR[m]} {y}" if fmt == "mon" else f"{y}"


def weighted(rng, pairs):
    r, acc = rng.random(), 0.0
    for v, w in pairs:
        acc += w
        if r <= acc:
            return v
    return pairs[-1][0]


def make_candidate(rng, arch_key, level, i, used_names):
    A = ARCH[arch_key]
    while True:
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name not in used_names:
            used_names.add(name)
            break
    core = list(A["core"])
    if arch_key == "mobile":
        core = ["Android", "Kotlin"] if i % 2 == 0 else ["iOS", "Swift"]
    extra = [e for e in A["extra"] if e not in _NOT_IN_TAXONOMY and e not in core]
    # de-duplicate (frontend/devops lists intentionally weight some skills)
    extra_w = list(extra)
    extra = list(dict.fromkeys(extra))

    total_y = rng.uniform(*LEVEL_YEARS[level])
    total_m = int(total_y * 12)
    n_jobs = 1 if level == "junior" else rng.choice([2, 3]) if level == "senior" else 2
    cuts = sorted(rng.sample(range(int(total_m * 0.2), int(total_m * 0.8)), n_jobs - 1)) if n_jobs > 1 else []
    bounds = [0] + cuts + [total_m]
    durations = [bounds[k + 1] - bounds[k] for k in range(n_jobs)]  # oldest -> newest

    jobs, end = [], NOW
    for k in range(n_jobs - 1, -1, -1):  # newest first
        start = end - durations[k]
        recency = (n_jobs - 1 - k)  # 0 = newest
        n_extra = rng.randint(2, 4) if recency == 0 else rng.randint(1, 3)
        pool = list(dict.fromkeys(rng.sample(extra_w, k=min(len(extra_w), n_extra + 2))))
        job_extras = [s for s in pool if s in extra][:n_extra]
        skills = core + [s for s in job_extras if s not in core]
        title = A["titles"][level] if recency == 0 else A["titles"]["junior" if level != "junior" else "junior"]
        if recency > 0 and level == "senior":
            title = A["titles"]["mid"]
        jobs.append({"title": title, "company": rng.choice(COMPANIES), "start": start,
                     "end": end, "present": recency == 0, "skills": skills})
        end = start - rng.randint(0, 3)
    jobs.reverse()  # oldest -> newest in storage, written newest first

    # job-bullet text
    for j in jobs:
        sk = list(j["skills"])
        rng.shuffle(sk)
        bullets = []
        pairs = [sk[x:x + 2] for x in range(0, len(sk), 2)]
        tmpl = list(A["flavor"]) + GENERIC_FLAVOR
        rng.shuffle(tmpl)
        for pi, pair in enumerate(pairs[:4]):
            a = surf(rng, pair[0])
            b = surf(rng, pair[1]) if len(pair) > 1 else a
            t = tmpl[pi % len(tmpl)]
            if "{b}" in t and len(pair) == 1:
                t = t.replace(" and {b}", "").replace("{b}", a)
            bullets.append(t.format(a=a, b=b, n=rng.randint(12, 60), t=rng.randint(3, 9)))
        # guarantee every ground-truth skill really appears in the job's text
        present = set(extract_skills(" ".join(bullets)))
        absent = [s for s in j["skills"] if s not in present]
        if absent:
            bullets.append("Day-to-day toolkit: " + ", ".join(surf(rng, s) for s in absent) + ".")
        j["bullets"] = bullets

    all_job_skills = list(dict.fromkeys(s for j in jobs for s in j["skills"]))
    familiar_pool = [s for s in extra if s not in all_job_skills]
    familiar = rng.sample(familiar_pool, k=min(len(familiar_pool), rng.randint(0, 2)))

    certs = []
    if arch_key in ("devops", "ml_engineer", "data_engineer", "product_manager") and rng.random() < 0.5:
        cands = [s for s in all_job_skills if s in CERTS]
        if cands:
            certs.append(CERTS[rng.choice(cands)])
    cert_skills = [s for s in CERTS if CERTS[s] in certs]

    edu_level = weighted(rng, [(2, 0.6), (3, 0.3), (4, 0.1 if arch_key in ("ml_engineer", "data_scientist") else 0.0)]) if level != "junior" else weighted(rng, [(2, 0.75), (3, 0.25)])
    if arch_key == "product_manager" and edu_level == 4:
        edu_level = 3
    field = weighted(rng, A["edu"])
    grad_year = NOW // 12 - int(total_y) - rng.randint(1, 3)
    uni = f"{rng.choice(['Eastern', 'Northern', 'Coastal', 'Central', 'Western'])} {rng.choice(UNIS)}"
    deg_text = {2: rng.choice(["B.S.", "Bachelor of Science", "B.Tech", "Bachelor of Engineering"]),
                3: rng.choice(["M.S.", "Master of Science", "MBA" if arch_key == "product_manager" else "M.Sc."]),
                4: "Ph.D."}[edu_level]
    education = []
    if edu_level >= 3:
        education.append(f"{deg_text} in {field} | {uni} | {grad_year}")
        education.append(f"{rng.choice(['B.S.', 'Bachelor of Science', 'B.Tech'])} in {field} | {uni} | {grad_year - rng.choice([2, 4])}")
    else:
        education.append(f"{deg_text} in {field} | {uni} | {grad_year}")

    projects = []
    if rng.random() < 0.35:
        ps = rng.choice(all_job_skills)
        projects.append(f"Open-source contributor: built a small tool using {surf(rng, ps)} with {rng.randint(40, 900)} GitHub stars.")

    # summary mentions only job skills (so ground truth stays exact)
    hi = rng.sample(all_job_skills, k=min(3, len(all_job_skills)))
    summary = (f"{A['titles'][level]} with {round(total_y)} years in industry building reliable, "
               f"user-focused solutions. Hands-on with {', '.join(surf(rng, s) for s in hi)}.")

    # ground truth ------------------------------------------------------------
    ivs = [(j["start"], j["end"]) for j in jobs]
    skill_iv = {}
    for j in jobs:
        for s in j["skills"]:
            skill_iv.setdefault(s, []).append((j["start"], j["end"]))
    skill_years = {s: round(months_in(v) / 12, 1) for s, v in skill_iv.items()}
    all_skills = list(dict.fromkeys(all_job_skills + familiar + cert_skills))
    for s in all_skills:
        skill_years.setdefault(s, 0.0)

    email = f"{name.lower().replace(' ', '.').replace('í', 'i').replace('á', 'a').replace('ú', 'u').replace('ó', 'o')}@example.com"
    return {
        "name": name, "email": email, "arch": arch_key, "level": level, "location": rng.choice(CITIES),
        "phone": f"+{rng.randint(1, 91)} {rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
        "summary": summary, "jobs": jobs, "familiar": familiar, "certs": certs, "projects": projects,
        "education": education,
        "truth": {"name": name, "archetype": arch_key, "level": level, "skills": all_skills,
                  "skill_years": skill_years, "total_years": round(months_in(ivs) / 12, 1),
                  "education_level": edu_level},
    }


def render_resume_text(rng, c) -> str:
    h = {k: rng.choice(v) for k, v in HEADINGS.items()}
    fmt = rng.choice(["mon", "mon", "mon", "year"])  # mostly "Jan 2019", sometimes "2019"
    dash = rng.choice([" - ", " \u2013 ", " - "])
    L = [c["name"].upper() if rng.random() < 0.3 else c["name"],
         f"{c['email']} | {c['phone']} | {c['location']}", ""]

    def summary():
        return [h["summary"], c["summary"], ""]

    def skills():
        sk = list(dict.fromkeys(s for j in c["jobs"] for s in j["skills"])) + c["familiar"]
        sk = [surf(rng, s) if rng.random() < 0.5 else s for s in sk]
        rng.shuffle(sk)
        if rng.random() < 0.5:
            half = len(sk) // 2
            return [h["skills"], "Core: " + ", ".join(sk[:half]), "Also: " + ", ".join(sk[half:]), ""]
        return [h["skills"], ", ".join(sk), ""]

    def experience():
        out = [h["experience"]]
        for j in sorted(c["jobs"], key=lambda j: -j["start"]):
            end = "Present" if j["present"] else idx_to_str(j["end"], fmt)
            out.append(f"{j['title']} | {j['company']} | {idx_to_str(j['start'], fmt)}{dash}{end}")
            out += [f"- {b}" for b in j["bullets"]]
        return out + [""]

    def education():
        return [h["education"]] + c["education"] + [""]

    def projects():
        return ([h["projects"]] + [f"- {p}" for p in c["projects"]] + [""]) if c["projects"] else []

    def certs():
        return ([h["certifications"]] + [f"- {x}" for x in c["certs"]] + [""]) if c["certs"] else []

    order = rng.choice([
        [summary, skills, experience, education, projects, certs],
        [summary, experience, skills, education, projects, certs],
        [summary, experience, education, skills, certs, projects],
    ])
    for fn in order:
        L += fn()
    return "\n".join(L).rstrip() + "\n"


def write_pdf(text: str, path: Path) -> None:
    from xml.sax.saxutils import escape
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=12.5)
    head = ParagraphStyle("h", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=6)
    title = ParagraphStyle("t", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=17, leading=21)
    known = {a.upper() for v in HEADINGS.values() for a in v}
    story = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            story.append(Spacer(1, 4))
        elif i == 0:
            story.append(Paragraph(escape(line), title))
        elif line.strip() in known:
            story.append(Paragraph(escape(line), head))
        else:
            story.append(Paragraph(escape(line), body))
    SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                      topMargin=0.7 * inch, bottomMargin=0.7 * inch, title="Resume").build(story)


# ------------------------------ job descriptions ----------------------------- #
JDS = [
    dict(slug="jd_01_senior_ml_engineer", primary=["ml_engineer"], secondary=["data_scientist"],
         must={"skills": {"Python": 5, "Machine Learning": 3}, "min_education": 2},
         text="""Senior Machine Learning Engineer

About the role
We build recommendation and ranking systems used by millions of customers. You will own models from prototype to production and work closely with data scientists and platform engineers.

Responsibilities
- Train, evaluate and deploy machine learning models at scale
- Build reliable serving and monitoring for model endpoints
- Mentor junior engineers and review designs

Requirements
- 5+ years of experience with Python
- 3+ years of experience with machine learning in production
- Hands-on experience with PyTorch or TensorFlow
- Experience with Docker and AWS
- Bachelor's degree in Computer Science or a related field

Nice to have
- Kubernetes and MLOps practices
- Experience with LLMs and RAG systems
- Spark for large-scale data processing
"""),
    dict(slug="jd_02_data_scientist", primary=["data_scientist"], secondary=["ml_engineer"],
         must={"skills": {"Python": 3, "SQL": 3}, "min_education": 2},
         text="""Data Scientist - Growth Analytics

About the role
Join our growth team to turn behavioural data into product decisions. You will design experiments, build predictive models and communicate insights to leadership.

Responsibilities
- Design and analyse controlled experiments
- Build churn and lifetime-value models
- Present recommendations to product and marketing leaders

Requirements
- 3+ years of experience with Python
- 3+ years of experience with SQL
- Strong foundation in statistics and A/B testing
- Experience with machine learning and scikit-learn
- Bachelor's degree in a quantitative field

Nice to have
- Master's degree preferred
- Tableau for dashboards
- Spark or Snowflake
"""),
    dict(slug="jd_03_data_engineer", primary=["data_engineer"], secondary=[],
         must={"skills": {"SQL": 4, "Spark": 3}, "min_education": 2},
         text="""Data Engineer - Platform

About the role
Our data platform powers analytics and machine-learning products across the company. You will design, build and operate large-scale batch and streaming workflows.

Responsibilities
- Build and maintain ingestion and transformation workflows
- Own data quality, lineage and cost efficiency
- Partner with analysts and scientists on data models

Requirements
- 4+ years of experience with SQL
- 3+ years of experience with Spark
- Experience with Airflow and Python
- Experience with AWS or GCP
- Bachelor's degree in Computer Science or equivalent

Nice to have
- Kafka and dbt
- Snowflake or BigQuery
"""),
    dict(slug="jd_04_backend_java_engineer", primary=["backend_java"], secondary=[],
         must={"skills": {"Java": 5, "Spring Boot": 3}},
         text="""Senior Backend Engineer (Java)

About the role
Our payments platform processes thousands of transactions per second. You will design resilient services and raise the bar on engineering quality.

Responsibilities
- Design, build and operate backend services
- Lead technical design reviews
- Improve performance, reliability and observability

Requirements
- 5+ years of experience with Java
- 3+ years of experience with Spring Boot
- Experience designing microservices and REST APIs
- Experience with PostgreSQL or MySQL

Nice to have
- Kafka
- Docker and Kubernetes
- Redis
"""),
    dict(slug="jd_05_frontend_engineer", primary=["frontend"], secondary=[],
         must={"skills": {"React": 3, "TypeScript": 3}},
         text="""Frontend Engineer - Web Platform

About the role
You will build the customer-facing web application used by hundreds of thousands of people each month, partnering closely with design and product.

Responsibilities
- Build accessible, performant user interfaces
- Maintain our shared component library
- Collaborate with backend engineers on API design

Requirements
- 3+ years of experience with React
- 3+ years of experience with TypeScript
- Solid HTML/CSS and JavaScript fundamentals
- Experience consuming REST APIs or GraphQL

Nice to have
- Node.js
- Figma and design-system experience
- CI/CD for frontend delivery
"""),
    dict(slug="jd_06_devops_sre", primary=["devops"], secondary=[],
         must={"skills": {"Kubernetes": 4, "Terraform": 3}},
         text="""Site Reliability Engineer / DevOps

About the role
Help us run a multi-region cloud platform with strong reliability targets. You will build the tooling and automation that lets product teams ship safely.

Responsibilities
- Operate and improve production infrastructure
- Build deployment automation and self-service tooling
- Lead incident response and post-incident reviews

Requirements
- 4+ years of experience with Kubernetes
- 3+ years of experience with Terraform
- Experience with AWS and Linux
- Strong CI/CD practices

Nice to have
- Prometheus and Grafana
- Python scripting
"""),
    dict(slug="jd_07_applied_llm_engineer", primary=["ml_engineer"], secondary=["data_scientist"],
         must={"skills": {"Python": 2, "LLMs": 1}},
         text="""Applied AI Engineer - LLM Applications

About the role
We are building customer-facing assistants powered by large language models. You will design retrieval-augmented generation pipelines, evaluate quality and ship to production.

Responsibilities
- Build and evaluate RAG systems end to end
- Fine-tune and prompt large language models
- Monitor quality, cost and latency in production

Requirements
- 2+ years of experience with Python
- 1+ years of experience with LLMs
- Experience with RAG and NLP
- Familiarity with Docker and AWS

Nice to have
- PyTorch
- FastAPI
- MLOps
"""),
    dict(slug="jd_08_product_manager", primary=["product_manager"], secondary=[],
         must={"skills": {"Product Management": 3}, "min_education": 2},
         text="""Product Manager - Data Products

About the role
Own the roadmap for our customer analytics product, working with engineering, design and go-to-market teams.

Responsibilities
- Define the vision and prioritise the backlog
- Run discovery with customers and translate needs into requirements
- Track outcomes and iterate

Requirements
- 3+ years of experience in product management
- Experience working in Agile teams
- Comfortable with SQL for self-serve analysis
- Bachelor's degree

Nice to have
- Tableau or Excel modelling
- A/B testing experience
- JIRA
"""),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    (out / "resumes").mkdir(parents=True, exist_ok=True)
    (out / "job_descriptions").mkdir(parents=True, exist_ok=True)

    used, truth, n = set(), {}, 0
    for arch_key, A in ARCH.items():
        for i in range(A["count"]):
            level = LEVEL_CYCLE[i % len(LEVEL_CYCLE)]
            c = make_candidate(rng, arch_key, level, i, used)
            text = render_resume_text(rng, c)
            n += 1
            stem = f"{n:02d}_{c['name'].lower().replace(' ', '_')}"
            stem = "".join(ch for ch in stem if ch.isascii())
            if rng.random() < 0.75:
                fn = f"{stem}.pdf"
                write_pdf(text, out / "resumes" / fn)
            else:
                fn = f"{stem}.txt"
                (out / "resumes" / fn).write_text(text, encoding="utf-8")
            truth[fn] = c["truth"]

    jd_truth = {}
    for jd in JDS:
        (out / "job_descriptions" / f"{jd['slug']}.txt").write_text(jd["text"], encoding="utf-8")
        jd_truth[jd["slug"]] = {"primary": jd["primary"], "secondary": jd["secondary"], "must_haves": jd["must"]}

    (out / "ground_truth.json").write_text(json.dumps({"resumes": truth, "job_descriptions": jd_truth}, indent=2))
    print(f"Wrote {len(truth)} resumes and {len(JDS)} job descriptions to {out}/")


if __name__ == "__main__":
    main()

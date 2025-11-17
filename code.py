import os
import re
from datetime import datetime
import pandas as pd
import math
import random
from collections import Counter
import sys
import time 

try:
    import spacy
    nlp = spacy.load("en_core_web_lg")
except OSError:
    os.system("python -m spacy download en_core_web_lg")
    import spacy
    nlp = spacy.load("en_core_web_lg")

EXP_PATTERN = re.compile(
    r"(?P<years>\d+(?:\.\d+)?)(?:\+|-)?\s*(?:years?|yrs?\.?)?\s*(?:of\s*)?experience|"
    r"(?P<start_month>[a-zA-Z]{3,9})?\s*(?P<start_year>\d{4})\s*[-to]+\s*(?P<end_month>[a-zA-Z]{3,9})?\s*(?P<end_year>\d{4})\s*(?:experience)?",
    re.IGNORECASE
)

SCORING_MAP = {
    "Machine Learning": 5, "Deep Learning": 5, "NLP": 4, "Computer Vision": 3,
    "Programming Languages": 2, "Cloud & Deployment": 2, "Certification": 2, "Data Science": 3,
    "Other Skills": 1,
    "Education Qualification": {
        "PhD": 5, "MTech": 4, "MCA": 4, "B.Tech": 3, "BCA": 2, "BSC": 2, "B.CA": 2
    },
    "Experience": {
        (0, 2.5): 2, (2.5, 5.5): 3, (5.5, float('inf')): 5
    }
}

KEYWORDS_DICT = {
    "Machine Learning": ["machine learning", "clustering", "logistic regression", "classification", "scikit learn",
                         "model selection", "model evaluation", "pyspark", "hadoop", "big data pipelines"],
    "Data Science": ["Junior Data Engineer", "Data Analysis", "Data Analytics", "Linear Regression",
                     "Predictive Modeling", "Business Analytics", "data modeling", "tableau"],
    "Deep Learning": ["deep learning", "tensorflow", "keras", "pytorch", "neural networks", "cnn", "rnn", "gans"],
    "NLP": ["natural language processing", "nlp", "nltk", "spacy", "beautiful soup", "transformers"],
    "Cloud & Deployment": ["aws", "azure", "google cloud platform", "firebase", "django", "docker", "kubernetes", "jenkins", "terraform", "ci/cd", "devops"],
    "Programming Languages": ["python", "r", "javascript", "node.js", "sql", "mongodb", "graphql", "numpy", "pandas"],
    "Education Qualification": ["phd", "mtech", "mca", "bca", "b.tech", "bsc", "B.CA"],
    "Other Skills": ["project manager", "pmp", "agile", "scrum", "ux/ui design", "react", "git", "selenium"]
}

SIMULATED_CANDIDATE_STATS = {
    1: (8.5, 0), 2: (6.2, 1), 3: (7.9, 0), 4: (9.1, 0), 5: (5.5, 2), 
    6: (7.0, 1), 7: (8.8, 0), 8: (6.5, 0), 9: (9.0, 0), 10: (5.9, 1),
    99: (7.5, 0)
}

MIN_CGPA = 6.0
MAX_BACKLOGS = 1

COMPANY_JOBS = {
    "TechCorp ML": {"focus": "Machine Learning", "min_exp": 2.0},
    "DataSolutions Inc.": {"focus": "Data Science", "min_exp": 1.0},
    "CloudStream DevOps": {"focus": "Cloud & Deployment", "min_exp": 3.0},
    "EntryLevel WebDev": {"focus": "Programming Languages", "min_exp": 0.5},
}

def read_pdf_or_docx_file(filepath):
    print(f"\n[SIMULATION] Reading text from '{filepath}'...")
    time.sleep(1)
    mock_resume_text = (
        "Experience: Data Scientist at DataWorks (Jan 2022 - Present). "
        "Built predictive models using Python, numpy, and pandas. "
        "Implemented deep learning models (Keras, TensorFlow) for NLP tasks. "
        "Deployed pipelines on AWS using Docker and Jenkins (CI/CD). "
        "Education: MTech in Computer Science."
    )
    if 'ML' in filepath:
        mock_resume_text += " Skills include scikit learn and classification."
    if 'Cloud' in filepath:
        mock_resume_text += " Expert in Kubernetes and Terraform."
    return mock_resume_text

def clean_resume_text(text):
    text = str(text)
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'&nbsp;|\s*&bull;\s*|&#\d+;|&amp;', ' ', clean, flags=re.IGNORECASE)
    clean = re.sub(r'id=""\w+""|itemprop=""\w+""|class=""\w+""', ' ', clean)
    clean = re.sub(r'jobdates|joblocation|jobcity|companyname|jobline|description', ' ', clean)
    clean = re.sub(r'[\u2022\u2023\u25E6\u2043\*\-]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def extract_years_of_experience(resume_text):
    total_years = 0.0
    current_year = datetime.now().year
    
    for match in EXP_PATTERN.finditer(resume_text):
        if match.group("years"):
            total_years += float(match.group("years"))
        elif match.group("start_year") and match.group("end_year"):
            start_year = int(match.group("start_year"))
            end_year_str = match.group("end_year")
            end_year = int(end_year_str) if end_year_str.isdigit() else current_year
            start_month = match.group("start_month") or "Jan"
            end_month = match.group("end_month") or "Dec"

            try:
                start_month = start_month.capitalize()[:3]
                end_month = end_month.capitalize()[:3]
                start_date = datetime.strptime(f"{start_month} {start_year}", "%b %Y")
                end_date = datetime.strptime(f"{end_month} {end_year}", "%b %Y")
                
                if end_date > start_date:
                    difference_in_years = (end_date - start_date).days / 365.25
                    total_years += difference_in_years
            except ValueError:
                pass
    return total_years

def match_experience_score(years_of_experience, experience_map):
    for (min_years, max_years), score in experience_map.items():
        if min_years <= years_of_experience < max_years:
            return score
    return 0

def score_resume(resume_text, keyword_dict, data_map):
    total_score = 0
    years_of_experience = extract_years_of_experience(resume_text)
    experience_score = match_experience_score(years_of_experience, data_map["Experience"])
    total_score += experience_score
    lower_text = resume_text.lower()
    scored_keywords = set()
    category_scores = {}
    
    for category in keyword_dict.keys():
        if category in data_map and category not in ["Experience", "Education Qualification"]:
            category_score = 0
            category_base_score = data_map[category]
            
            for keyword in keyword_dict[category]:
                lower_keyword = keyword.lower()
                if lower_keyword in lower_text:
                    if lower_keyword not in scored_keywords:
                        scored_keywords.add(lower_keyword)
                        category_score += category_base_score
                        total_score += category_base_score
                        
            category_scores[category] = category_score
            
    edu_score = 0
    for keyword in keyword_dict.get("Education Qualification", []):
        lower_keyword = keyword.lower()
        if lower_keyword in lower_text:
            edu_score = data_map["Education Qualification"].get(keyword, 0)
            total_score += edu_score
            break
    category_scores["Education Qualification"] = edu_score
    
    return total_score, years_of_experience, category_scores

def calculate_similarity(resume_text_a, resume_text_b):
    doc_a = nlp(resume_text_a)
    doc_b = nlp(resume_text_b)
    try:
        return doc_a.similarity(doc_b)
    except ValueError:
        return 0.0

def hard_filter_candidate(candidate_id, stats_map, min_cgpa, max_backlogs):
    stats = stats_map.get(candidate_id, (10.0, 0))
    cgpa, backlogs = stats
    is_eligible = (cgpa >= min_cgpa) and (backlogs <= max_backlogs)
    return is_eligible, cgpa, backlogs

def generate_improvement_feedback(candidate_data, target_category):
    years_exp = candidate_data['years_exp']
    target_score = candidate_data['category_scores'].get(target_category, 0)
    
    feedback = f"🎯 **Target Role: {target_category}**\n"
    
    if target_score == 0:
        feedback += "❌ Low skill evidence. Add relevant projects, tools (e.g., TensorFlow, Kubernetes), and courses.\n"
    elif target_score < 10:
        feedback += "⚠️ Decent start. Include 1-2 more key projects or certifications related to this field.\n"
    else:
        feedback += "✅ Strong skill set! Ensure your most impressive project is detailed prominently.\n"
        
    min_exp_req = COMPANY_JOBS.get(target_category, {}).get("min_exp", 1.0)
    if years_exp < min_exp_req:
          feedback += f"💡 Note: Your {years_exp:.1f} yrs experience is below the typical {min_exp_req:.1f} yr minimum for this role. Highlight internships.\n"
    
    return feedback.strip()

def smart_match_to_companies(candidate_data, company_jobs):
    fit_scores = {}
    
    for company_name, requirements in company_jobs.items():
        focus_category = requirements.get("focus")
        min_exp = requirements.get("min_exp", 0)
        
        category_score = candidate_data['category_scores'].get(focus_category, 0)
        
        exp_factor = 1.0
        if candidate_data['years_exp'] < min_exp:
            exp_factor = 0.5 if candidate_data['years_exp'] >= min_exp * 0.5 else 0.1
        
        fit_score = category_score * exp_factor
        fit_scores[company_name] = fit_score
        
    best_matches = sorted(fit_scores.items(), key=lambda item: item[1], reverse=True)
    return best_matches

def cluster_students_by_skills(resumes_data):
    clusters = {
        "ML/DL/NLP Specialist": [],
        "Data Science/Analytics Expert": [],
        "Cloud/DevOps Engineer": [],
        "Generalist/Other": []
    }
    
    for name, data in resumes_data.items():
        category_scores = data['category_scores']
        
        primary_scores = {
            "ML/DL/NLP Specialist": category_scores.get("Machine Learning", 0) + category_scores.get("Deep Learning", 0) + category_scores.get("NLP", 0),
            "Data Science/Analytics Expert": category_scores.get("Data Science", 0),
            "Cloud/DevOps Engineer": category_scores.get("Cloud & Deployment", 0),
        }
        
        if not primary_scores or max(primary_scores.values()) == 0:
            assigned_cluster = "Generalist/Other"
        else:
            max_score = max(primary_scores.values())
            assigned_cluster = next((k for k, v in primary_scores.items() if v == max_score), "Generalist/Other")
            
        clusters[assigned_cluster].append(name)
        data['cluster'] = assigned_cluster
        
    return clusters

def format_table(title, data, headers):
    print(f"\n--- {title} ---")
    
    col_widths = {
        'Candidate Name': 25, 'Total Score': 12, 'Years Experience': 18, 
        'ML Score': 10, 'DS Score': 10, 'Cloud Score': 12, 'Eligible': 10,
        'Candidate A': 25, 'Candidate B': 25, 'Similarity Score': 18, 
        'Company': 25, 'Fit Score': 12, 'Cluster': 30, 'CGPA': 8, 'Backlogs': 10
    }

    current_widths = [col_widths.get(h, len(h) + 2) for h in headers]
    
    def pad_str(s, width):
        s = str(s).strip()
        s = re.sub(r'\s+', ' ', s)
        if len(s) > width:
            return s[:width - 3] + "..."
        return s.ljust(width)
    
    separator = "+" + "+".join(["-" * w for w in current_widths]) + "+"
    print(separator)
    
    header_line = "|" + "|".join([pad_str(h, current_widths[i]) for i, h in enumerate(headers)]) + "|"
    print(header_line)
    print(separator)
    
    for row in data:
        row_line = "|" + "|".join([pad_str(item, current_widths[i]) for i, item in enumerate(row)]) + "|"
        print(row_line)
    
    print(separator)

def display_ranked_candidates(resumes_data, category=None, top_n=5):
    if not resumes_data:
        print("No eligible candidates to display.")
        return

    if category is None:
        title = f"Top {top_n} Overall Candidates (Shortlisted by Total Score)"
        headers = ['Candidate Name', 'Total Score', 'Years Experience', 'Cluster', 'ML Score', 'DS Score', 'Cloud Score']
        
        ranked_list = sorted(
            [(name, data) for name, data in resumes_data.items()],
            key=lambda x: x[1]['score'], reverse=True
        )[:top_n]
        
        data_to_display = [
            (name, 
             f"{data['score']:.2f}",
             f"{data['years_exp']:.2f}",
             data.get('cluster', 'N/A'),
             f"{data['ml_score']:.2f}",
             f"{data['ds_score']:.2f}",
             f"{data['cloud_score']:.2f}")
            for name, data in ranked_list
        ]
    else:
        title = f"Top {top_n} Candidates for **{category}**"
        key_map = {'ML': 'ml_score', 'DS': 'ds_score', 'Cloud/DevOps': 'cloud_score'}
        sort_key = key_map.get(category, 'score')
        
        headers = ['Candidate Name', f'{category} Score', 'Years Experience', 'Cluster']
        
        ranked_list = sorted(
            [(name, data) for name, data in resumes_data.items()],
            key=lambda x: x[1].get(sort_key, 0), reverse=True
        )[:top_n]
        
        data_to_display = [
            (name, 
             f"{data.get(sort_key, 0):.2f}",
             f"{data['years_exp']:.2f}",
             data.get('cluster', 'N/A'))
            for name, data in ranked_list
        ]

    format_table(title, data_to_display, headers)

def display_similarity_matching(resumes_data, top_n=5):
    if len(resumes_data) < 2:
        print("Need at least two eligible candidates for similarity analysis.")
        return
        
    ranked_candidates = sorted(
        [(name, data) for name, data in resumes_data.items()],
        key=lambda x: x[1]['score'], reverse=True
    )
    
    top_names = [name for name, _ in ranked_candidates[:top_n]]
    similarity_data = []

    for i in range(len(top_names)):
        for j in range(i + 1, len(top_names)):
            name_a = top_names[i]
            name_b = top_names[j]
            
            text_a = resumes_data[name_a]['text']
            text_b = resumes_data[name_b]['text']
            
            similarity = calculate_similarity(text_a, text_b)
            
            similarity_data.append((name_a, name_b, f"{similarity:.4f}"))

    format_table(
        f"Resume Pair Similarity (Cosine, Top {top_n} Overall)",
        similarity_data,
        ['Candidate A', 'Candidate B', 'Similarity Score']
    )

def display_feedback_and_company_matching(resumes_data, company_jobs):
    if not resumes_data:
        print("No eligible candidates to analyze.")
        return

    ranked_candidates = sorted(
        [(name, data) for name, data in resumes_data.items()],
        key=lambda x: x[1]['score'], reverse=True
    )
    
    top_3_names = [name for name, _ in ranked_candidates[:min(3, len(ranked_candidates))]]
    
    match_data_display = []
    
    print("\n" + "="*80)
    print("                  SMART MATCHING AND IMPROVEMENT FEEDBACK")
    print("="*80)

    for candidate_name in top_3_names:
        data = resumes_data[candidate_name]
        
        best_matches = smart_match_to_companies(data, company_jobs)
        
        match_data_display.append((candidate_name, best_matches[0][0], f"{best_matches[0][1]:.2f}", data.get('cluster', 'N/A')))
        
        print(f"\n\n--- 🗣️ Improvement Feedback for **{candidate_name}** ---")
        best_focus = company_jobs[best_matches[0][0]]['focus']
        print(generate_improvement_feedback(data, best_focus))
        print("-" * 30)

    format_table(
        "Best Company Match for Top Candidates",
        match_data_display,
        ['Candidate Name', 'Company', 'Fit Score', 'Cluster']
    )

def display_clustering(resumes_data):
    if not resumes_data:
        print("No eligible candidates to cluster.")
        return

    clusters = cluster_students_by_skills(resumes_data)
    
    cluster_data_display = []
    for cluster_name, members in clusters.items():
        if members:
            for member in members:
                cluster_data_display.append((member, cluster_name))
    
    cluster_data_display.sort(key=lambda x: x[1])

    format_table(
        "AI-Based Student Skill Clustering",
        cluster_data_display,
        ['Candidate Name', 'Cluster']
    )

def display_initial_shortlisting(df_resumes, resumes_data, stats_map, min_cgpa, max_backlogs):
    all_data = []
    
    for index, row in df_resumes.iterrows():
        candidate_id = row.get('ID', index + 1)
        candidate_name = f"Candidate_{candidate_id}"
        
        is_eligible, cgpa, backlogs = hard_filter_candidate(candidate_id, stats_map, min_cgpa, max_backlogs)
        
        if candidate_name in resumes_data:
            data = resumes_data[candidate_name]
            total_score = f"{data['score']:.2f}"
            years_exp = f"{data['years_exp']:.2f}"
        else:
            total_score = "N/A"
            years_exp = "N/A"

        all_data.append((
            candidate_name,
            total_score,
            years_exp,
            f"{cgpa:.2f}",
            str(backlogs),
            "YES" if is_eligible else "NO"
        ))

    all_data.sort(key=lambda x: x[5] == "NO")

    format_table(
        f"Automated Resume Shortlisting & Hard Filtering (Min CGPA: {min_cgpa}, Max Backlogs: {max_backlogs})",
        all_data,
        ['Candidate Name', 'Total Score', 'Years Experience', 'CGPA', 'Backlogs', 'Eligible']
    )

def display_candidate_workflow(resumes_data, stats_map):
    print("\n" + "="*80)
    print("                        NEW CANDIDATE WORKFLOW")
    print("="*80)

    file_type = input("Enter file type for simulation (PDF or DOCX): ").upper()
    
    if file_type not in ['PDF', 'DOCX']:
        print("❌ Invalid file type. Please enter PDF or DOCX.")
        return

    file_path = f"resume_{file_type}_ML.pdf"
    raw_text = read_pdf_or_docx_file(file_path)
    clean_text = clean_resume_text(raw_text)
    
    candidate_id = 99
    candidate_name = f"Candidate_{candidate_id}"

    is_eligible, cgpa, backlogs = hard_filter_candidate(candidate_id, stats_map, MIN_CGPA, MAX_BACKLOGS)
    
    print(f"\n[STEP 2: Hard Filtering] CGPA: {cgpa:.2f}, Backlogs: {backlogs}")
    if not is_eligible:
        print(f"❌ Result: INELIGIBLE. Candidate {candidate_name} failed the CGPA/Backlog check.")
        return
    print("✅ Result: ELIGIBLE. Proceeding to scoring.")

    total_score, years_of_experience, category_scores = score_resume(clean_text, KEYWORDS_DICT, SCORING_MAP)
    
    new_candidate_data = {
        'id': candidate_id,
        'score': total_score,
        'years_exp': round(years_of_experience, 2),
        'text': clean_text,
        'category_scores': category_scores,
        'ml_score': category_scores.get('Machine Learning', 0),
        'ds_score': category_scores.get('Data Science', 0),
        'cloud_score': category_scores.get('Cloud & Deployment', 0),
        'cgpa': cgpa, 'backlogs': backlogs
    }
    
    temp_cluster_data = {candidate_name: new_candidate_data}
    cluster = cluster_students_by_skills(temp_cluster_data)
    assigned_cluster = new_candidate_data.get('cluster', list(cluster.keys())[0] if cluster else 'N/A')
    new_candidate_data['cluster'] = assigned_cluster
    
    print(f"\n[STEP 3 & 4: Scoring] Total Score: {total_score:.2f} | Years Exp: {years_of_experience:.2f}")
    print(f"[STEP 5: Clustering] Assigned Cluster: {assigned_cluster}")
    
    best_matches = smart_match_to_companies(new_candidate_data, COMPANY_JOBS)
    best_focus = COMPANY_JOBS[best_matches[0][0]]['focus']
    feedback = generate_improvement_feedback(new_candidate_data, best_focus)

    print(f"\n[STEP 6: Smart Matching] Best Match: {best_matches[0][0]} (Fit Score: {best_matches[0][1]:.2f})")
    
    print(f"\n[STEP 8: Personalized Feedback for {candidate_name}]")
    print("-" * 50)
    print(feedback)
    print("-" * 50)

    resumes_data[candidate_name] = new_candidate_data
    print(f"\n✅ Candidate {candidate_name} processed and added to the ranking system.")

def menu_driven_main():
    file_path = "12.csv"
    text_column = "Resume_str"

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found. Please place it in the directory.")
        return
        
    try:
        df_resumes = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    if text_column not in df_resumes.columns:
        print(f"Error: The required column '{text_column}' was not found in the CSV.")
        return

    resumes_data = {}
    
    for index, row in df_resumes.iterrows():
        raw_text = row[text_column]
        candidate_id = row.get('ID', index + 1)
        candidate_name = f"Candidate_{candidate_id}"
        
        is_eligible, _, _ = hard_filter_candidate(candidate_id, SIMULATED_CANDIDATE_STATS, MIN_CGPA, MAX_BACKLOGS)
        clean_text = clean_resume_text(raw_text)
        
        if is_eligible and clean_text:
            total_score, years_of_experience, category_scores = score_resume(clean_text, KEYWORDS_DICT, SCORING_MAP)
            
            resumes_data[candidate_name] = {
                'id': candidate_id,
                'score': total_score,
                'years_exp': round(years_of_experience, 2),
                'text': clean_text,
                'category_scores': category_scores,
                'ml_score': category_scores.get('Machine Learning', 0),
                'ds_score': category_scores.get('Data Science', 0),
                'cloud_score': category_scores.get('Cloud & Deployment', 0),
            }
            
    cluster_students_by_skills(resumes_data)
    
    if not resumes_data:
        print("\n🚫 No candidates were eligible after filtering and cleaning. Cannot proceed to menu.")
        return

    while True:
        print("\n" + "="*60)
        print("        RESUME SHORTLISTING & MATCHING SYSTEM")
        print("="*60)
        print("1. Automated Resume Shortlisting (All Candidates)")
        print("2. **Candidate Workflow:** Simulate PDF/DOCX Upload")
        print("3. Ranking by Total Score")
        print("4. Ranking by Specific Skill Category (ML/DS/Cloud)")
        print("5. AI-Based Clustering of Students")
        print("6. Smart Matching to Companies & Feedback Generation")
        print("7. Resume-to-Resume Similarity Matching")
        print("0. Exit")
        print("-" * 60)
        
        choice = input("Enter your choice (0-7): ")
        
        try:
            choice = int(choice)
        except ValueError:
            print("\n❌ Invalid input. Please enter a number.")
            continue
            
        if choice == 1:
            display_initial_shortlisting(df_resumes, resumes_data, SIMULATED_CANDIDATE_STATS, MIN_CGPA, MAX_BACKLOGS)
        elif choice == 2:
            display_candidate_workflow(resumes_data, SIMULATED_CANDIDATE_STATS)
        elif choice == 3:
            display_ranked_candidates(resumes_data, category=None, top_n=10)
        elif choice == 4:
            print("\n--- Select Skill Category for Ranking ---")
            print("a. ML (Machine Learning)")
            print("b. DS (Data Science)")
            print("c. Cloud/DevOps")
            category_choice = input("Enter category (a/b/c): ").lower()
            if category_choice == 'a':
                display_ranked_candidates(resumes_data, category='ML', top_n=5)
            elif category_choice == 'b':
                display_ranked_candidates(resumes_data, category='DS', top_n=5)
            elif category_choice == 'c':
                display_ranked_candidates(resumes_data, category='Cloud/DevOps', top_n=5)
            else:
                print("\n❌ Invalid category choice.")
        elif choice == 5:
            display_clustering(resumes_data)
        elif choice == 6:
            display_feedback_and_company_matching(resumes_data, COMPANY_JOBS)
        elif choice == 7:
            display_similarity_matching(resumes_data, top_n=5)
        elif choice == 0:
            print("\n👋 Exiting the Resume System. Goodbye!")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please select from the menu options.")

            
if __name__ == '__main__':
    menu_driven_main()

import pandas as pd
import numpy as np
from datetime import datetime
import random

def generate_student_data(n_students=5000, random_state=42):
    """
    Generate synthetic student data for performance prediction
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Student demographics
    student_ids = [f"STU_{i:06d}" for i in range(n_students)]
    genders = np.random.choice(['M', 'F'], n_students, p=[0.52, 0.48])
    school_types = np.random.choice(['Public', 'Private', 'Charter'], n_students, p=[0.6, 0.3, 0.1])
    
    # Parent education levels
    parent_edu = np.random.choice(['High School', 'Some College', 'Bachelor', 'Master', 'PhD'], 
                                 n_students, p=[0.25, 0.30, 0.25, 0.15, 0.05])
    
    # Academic background
    prior_gpa = np.random.normal(3.2, 0.6, n_students)
    prior_gpa = np.clip(prior_gpa, 0.0, 4.0)
    
    # Current semester performance factors
    attendance_pct = np.random.normal(85, 12, n_students)
    attendance_pct = np.clip(attendance_pct, 40, 100)
    
    study_hours_wk = np.random.gamma(2, 3, n_students)
    study_hours_wk = np.clip(study_hours_wk, 1, 40)
    
    commute_min = np.random.exponential(25, n_students)
    commute_min = np.clip(commute_min, 5, 120)
    
    # Academic performance metrics
    quiz_avg = np.random.normal(75, 15, n_students)
    quiz_avg = np.clip(quiz_avg, 20, 100)
    
    assign_avg = np.random.normal(78, 12, n_students)
    assign_avg = np.clip(assign_avg, 30, 100)
    
    midterm = np.random.normal(72, 18, n_students)
    midterm = np.clip(midterm, 25, 100)
    
    # Engagement metrics
    on_time_submit_pct = np.random.beta(8, 2, n_students) * 100
    lms_logins_wk = np.random.poisson(4, n_students) + 1
    forum_posts = np.random.poisson(2, n_students)
    
    # Create DataFrame
    df = pd.DataFrame({
        'student_id': student_ids,
        'gender': genders,
        'school_type': school_types,
        'parent_edu': parent_edu,
        'prior_gpa': prior_gpa,
        'attendance_pct': attendance_pct,
        'study_hours_wk': study_hours_wk,
        'commute_min': commute_min,
        'quiz_avg': quiz_avg,
        'assign_avg': assign_avg,
        'midterm': midterm,
        'on_time_submit_pct': on_time_submit_pct,
        'lms_logins_wk': lms_logins_wk,
        'forum_posts': forum_posts
    })
    
    # Generate final performance based on weighted factors
    # This creates realistic correlations between features and outcomes
    performance_weights = {
        'prior_gpa': 0.25,
        'attendance_pct': 0.15,
        'study_hours_wk': 0.10,
        'quiz_avg': 0.15,
        'assign_avg': 0.15,
        'midterm': 0.20
    }
    
    # Calculate performance score
    df['performance_score'] = (
        df['prior_gpa'] * 25 * performance_weights['prior_gpa'] +
        df['attendance_pct'] * performance_weights['attendance_pct'] +
        df['study_hours_wk'] * 2.5 * performance_weights['study_hours_wk'] +
        df['quiz_avg'] * performance_weights['quiz_avg'] +
        df['assign_avg'] * performance_weights['assign_avg'] +
        df['midterm'] * performance_weights['midterm']
    )
    
    # Add some noise and demographic factors
    df['performance_score'] += np.random.normal(0, 5, n_students)
    
    # School type adjustment (private schools tend to have slightly better outcomes)
    df.loc[df['school_type'] == 'Private', 'performance_score'] += 3
    df.loc[df['school_type'] == 'Charter', 'performance_score'] += 1
    
    # Parent education adjustment
    edu_boost = {'High School': -2, 'Some College': 0, 'Bachelor': 2, 'Master': 4, 'PhD': 5}
    for edu, boost in edu_boost.items():
        df.loc[df['parent_edu'] == edu, 'performance_score'] += boost
    
    # Final score and grade
    df['final_score'] = np.clip(df['performance_score'], 0, 100)
    
    # Grade bands
    def get_grade_band(score):
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
    
    df['final_grade_band'] = df['final_score'].apply(get_grade_band)
    df['passed'] = (df['final_score'] >= 60).astype(int)
    
    # Drop intermediate column
    df = df.drop('performance_score', axis=1)
    
    # Reorder columns
    column_order = [
        'student_id', 'gender', 'school_type', 'parent_edu', 'prior_gpa',
        'attendance_pct', 'study_hours_wk', 'commute_min', 'quiz_avg', 
        'assign_avg', 'midterm', 'on_time_submit_pct', 'lms_logins_wk', 
        'forum_posts', 'final_score', 'final_grade_band', 'passed'
    ]
    
    df = df[column_order]
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic student dataset...")
    df = generate_student_data(n_students=5000)
    
    # Save to CSV and Parquet
    df.to_csv('data/students.csv', index=False)
    df.to_parquet('data/students.parquet', index=False)
    
    print(f"Dataset generated with {len(df)} students")
    print(f"Pass rate: {df['passed'].mean():.2%}")
    print(f"Grade distribution:")
    print(df['final_grade_band'].value_counts().sort_index())
    
    # Display sample
    print("\nSample data:")
    print(df.head())

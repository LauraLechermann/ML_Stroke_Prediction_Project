# =============================================================================
# STROKE RISK PREDICTION - FUNCTIONS MODULE
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display, Markdown, HTML
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, accuracy_score,
                           precision_score, recall_score, f1_score)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display, Markdown, HTML
import warnings
warnings.filterwarnings('ignore')


COLORS = {
    'primary': '#1f77b4',    
    'secondary': '#ff7f0e',   
    'success': '#2ca02c',    
    'warning': '#d62728',    
    'danger': '#9467bd',     
    'light': '#F1F1F1',       
    'dark': '#2D3436'          
}

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette([COLORS['primary'], COLORS['secondary'], COLORS['warning'], COLORS['success']])



# =============================================================================
# 1. DATA INSPECTION FUNCTIONS
# =============================================================================

def data_overview(df):
    """
    Provides comprehensive overview of the dataset
    
    Parameters:
    df (DataFrame): Input dataset
    """
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    

    column_info = pd.DataFrame({
        'Column': df.columns,
        'Data_Type': [str(dtype) for dtype in df.dtypes],
        'Unique_Values': [df[col].nunique() for col in df.columns],
        'Non_Null_Count': [df[col].count() for col in df.columns],
    })
    
    print("\n🔍 Column Information:")
    display(column_info)
    
    print("\n👀 First 5 Rows:")
    display(df.head())
    
    print("\n📈 Basic Statistics:")
    display(df.describe())
    
    if 'stroke' in df.columns:
        stroke_counts = df['stroke'].value_counts()
        stroke_pct = df['stroke'].value_counts(normalize=True) * 100
        print(f"\n🎯 Target Variable (Stroke) Distribution:")
        print(f"No Stroke (0): {stroke_counts[0]} ({stroke_pct[0]:.1f}%)")
        print(f"Stroke (1): {stroke_counts[1]} ({stroke_pct[1]:.1f}%)")
        print(f"Imbalance Ratio: 1:{stroke_counts[0]/stroke_counts[1]:.1f}")



def analyze_unique_values(df):
    """
    Creates a comprehensive table of unique values for each column
    
    Parameters:
    df (DataFrame): Input dataset
    """
    
    unique_info = []
    
    for col in df.columns:
        unique_vals = df[col].unique()
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        
    
        if unique_count <= 10:
            if df[col].dtype == 'object':
                # Sort string values
                sorted_vals = sorted([str(x) for x in unique_vals if pd.notna(x)])
                unique_str = ', '.join(sorted_vals)
            else:
                # For numeric, show sorted values
                sorted_vals = sorted([x for x in unique_vals if pd.notna(x)])
                unique_str = ', '.join([str(int(x)) if isinstance(x, (int, np.integer)) 
                                       else f"{x:.1f}" for x in sorted_vals])
        else:
            if df[col].dtype == 'object':
                sample_vals = list(unique_vals[:3])
                unique_str = f"{sample_vals} ... ({unique_count} total)"
            else:
                min_val = df[col].min()
                max_val = df[col].max()
                unique_str = f"{min_val:.1f} to {max_val:.1f} ({unique_count} values)"
        
        unique_info.append({
            'Column': col,
            'Data_Type': str(df[col].dtype),
            'Unique_Count': unique_count,
            'Null_Count': null_count,
            'Sample_Values': unique_str
        })
    
    unique_df = pd.DataFrame(unique_info)
    display(unique_df)
    

    print("\n🔍 Data Type Observations:")
    
    categorical_candidates = unique_df[
        (unique_df['Unique_Count'] <= 10) & 
        (unique_df['Data_Type'].isin(['int64', 'float64']))
    ]['Column'].tolist()
    
    if categorical_candidates:
        print(f"• Potential categorical variables (≤10 unique values): {categorical_candidates}")
    
    missing_cols = unique_df[unique_df['Null_Count'] > 0]['Column'].tolist()
    if missing_cols:
        print(f"• Columns with missing values: {missing_cols}")



# =============================================================================
# 2. DATA QUALITY CHECK FUNCTIONS
# =============================================================================

def check_missing_values(df):
    """
    Comprehensive analysis of missing values in the dataset
    
    Parameters:
    df (DataFrame): Input dataset
    """
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) == 0:
        print("✅ No missing values found in the dataset!")
    else:
        print(f"⚠️ Found {len(missing_df)} columns with missing values:\n")
        display(missing_df)
        
        plt.figure(figsize=(10, 6))
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
        
        ax = missing_pct.plot(kind='barh', color=COLORS['warning'])
        plt.xlabel('Missing Percentage (%)')
        plt.title('Missing Values by Column')
        plt.tight_layout()

        for i, v in enumerate(missing_pct):
            ax.text(v + 0.1, i, f'{v:.1f}%', va='center')
        
        plt.show()


        
def check_zero_values(df):
    """
    Check for zero values in numerical columns where zeros might be problematic
    
    Parameters:
    df (DataFrame): Input dataset
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    zero_info = []
    for col in numerical_cols:
        zero_count = (df[col] == 0).sum()
        zero_pct = (zero_count / len(df)) * 100
        
        if zero_count > 0:
            zero_info.append({
                'Column': col,
                'Zero_Count': zero_count,
                'Zero_Percentage': zero_pct,
                'Non_Zero_Count': len(df) - zero_count
            })
    
    if zero_info:
        zero_df = pd.DataFrame(zero_info).sort_values('Zero_Count', ascending=False)
        print("🔍 Zero Values in Numerical Columns:\n")
        display(zero_df)
        
        if 'bmi' in zero_df['Column'].values:
            print("• BMI: Zero values are medically impossible")
        if 'avg_glucose_level' in zero_df['Column'].values:
            print("• Average Glucose Level: Zero values are medically impossible")
        if 'age' in zero_df['Column'].values:
            print("• Age: Zero values are invalid")
    else:
        print("✅ No zero values found in numerical columns!")



def check_duplicates(df):
    """
    Check for duplicate rows in the dataset
    
    Parameters:
    df (DataFrame): Input dataset
    """
    duplicates = df.duplicated()
    duplicate_count = duplicates.sum()
    
    print(f"🔍 Duplicate Analysis:")
    print(f"• Total duplicate rows: {duplicate_count}")
    print(f"• Percentage of duplicates: {(duplicate_count/len(df))*100:.2f}%")
    
    if duplicate_count > 0:
        print("\n📋 Duplicate Row Details:")
        duplicate_indices = df[duplicates].index.tolist()
        print(f"• Duplicate row indices: {duplicate_indices[:10]}{'...' if len(duplicate_indices) > 10 else ''}")
        
        if 'stroke' in df.columns:
            dup_stroke_analysis = df[duplicates].groupby('stroke').size()
            print(f"\n• Stroke distribution in duplicates:")
            for stroke_val, count in dup_stroke_analysis.items():
                print(f"  - Stroke={stroke_val}: {count} duplicates")
        
        print("\n⚠️ Important: Do NOT remove duplicates before train-test split!")
        print("   Remove duplicates from training set only after splitting.")
    else:
        print("✅ No duplicate rows found!")




def check_data_types(df):
    """
    Verify data types and suggest corrections
    
    Parameters:
    df (DataFrame): Input dataset
    """
    print("📊 Data Type Analysis:\n")
    
    type_issues = []
    
    for col in df.columns:
        current_type = str(df[col].dtype)
        unique_count = df[col].nunique()
        
        # Check for potential type mismatches
        if col in ['hypertension', 'heart_disease', 'stroke'] and current_type != 'int64':
            type_issues.append(f"• {col}: Should be integer (binary), currently {current_type}")
        
        elif col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'] and current_type != 'object':
            type_issues.append(f"• {col}: Should be object (categorical), currently {current_type}")
        
        elif current_type in ['int64', 'float64'] and unique_count <= 5 and col not in ['stroke', 'hypertension', 'heart_disease']:
            type_issues.append(f"• {col}: Only {unique_count} unique values - consider treating as categorical")
    
    if type_issues:
        print("⚠️ Potential data type issues found:")
        for issue in type_issues:
            print(issue)
    else:
        print("✅ All data types appear correct!")
    
    type_summary = pd.DataFrame({
        'Column': df.columns,
        'Current_Type': df.dtypes,
        'Unique_Values': [df[col].nunique() for col in df.columns],
        'Sample_Value': [df[col].iloc[0] for col in df.columns]
    })
    
    print("\n📋 Data Type Summary:")
    display(type_summary)



def check_class_distribution(df, target_col):
    """
    Analyze the distribution of the target variable
    
    Parameters:
    df (DataFrame): Input dataset
    target_col (str): Name of target column
    """
    print(f"\n🎯 Target Variable Distribution: {target_col}")
    
    class_counts = df[target_col].value_counts()
    class_pct = df[target_col].value_counts(normalize=True) * 100
    
    distribution_df = pd.DataFrame({
        'Class': class_counts.index,
        'Count': class_counts.values,
        'Percentage': class_pct.values
    })
    
    display(distribution_df)
    
    majority_class = class_counts.iloc[0]
    minority_class = class_counts.iloc[1]
    imbalance_ratio = majority_class / minority_class
    
    print(f"\n📊 Imbalance Ratio: 1:{imbalance_ratio:.1f}")
    print(f"The majority class has {imbalance_ratio:.1f}x more samples than the minority class")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(class_counts.index.astype(str), class_counts.values, 
            color=[COLORS['primary'], COLORS['warning']])
    ax1.set_xlabel(target_col)
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution')
    
    for i, v in enumerate(class_counts.values):
        ax1.text(i, v + 50, str(v), ha='center')
    
    ax2.pie(class_counts.values, labels=['No Stroke', 'Stroke'], 
            autopct='%1.1f%%', colors=[COLORS['primary'], COLORS['warning']])
    ax2.set_title('Class Distribution (%)')
    
    plt.tight_layout()
    plt.show()
    
    print("\n⚠️ Highly imbalanced dataset - will need to address this during modeling!")



# =============================================================================
# 3. BASELINE MODEL FUNCTIONS
# =============================================================================

def evaluate_baseline(X_train, y_train, X_test, y_test, strategy="most_frequent", verbose=True):
    """
    Trains and evaluates a DummyClassifier as a baseline.
    
    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Test data
    - strategy (str): Strategy for DummyClassifier ("most_frequent", "stratified", "uniform")
    - verbose (bool): Whether to print results
    
    Returns:
    - metrics (dict): Dictionary with accuracy, recall, precision, and F1
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    
    dummy = DummyClassifier(strategy=strategy, random_state=42)
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics = {
        "Strategy": strategy,
        "Accuracy": acc,
        "Recall": recall,
        "Precision": precision,
        "F1 Score": f1
    }
    
    if verbose:
        print(f"\n📊 Baseline Model Evaluation (strategy='{strategy}'):")
        for k, v in metrics.items():
            if k != "Strategy":
                print(f"{k}: {v:.4f}")
    
    return metrics


# =============================================================================
# 4. PREPROCESSING FUNCTIONS
# =============================================================================

def visualize_missing_heatmap(df_train):
    """
    Visualize missing data patterns with heatmap
    """
    missing_matrix = df_train.isnull()
    
    if missing_matrix.sum().sum() == 0:
        print("✅ No missing values to visualize!")
        return
    
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(missing_matrix, 
                cbar=True,
                yticklabels=False,
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Missing Data'})
    
    plt.title('Missing Data Heatmap\n(Red = Missing, Blue = Present)', fontsize=14)
    plt.xlabel('Features')
    plt.ylabel('Samples')
    plt.tight_layout()
    plt.show()
    
    print("📊 Missing Data Summary:")
    for col in df_train.columns:
        missing_count = df_train[col].isnull().sum()
        if missing_count > 0:
            print(f"- {col}: {missing_count} missing ({missing_count/len(df_train)*100:.1f}%)")




def analyze_missing_vs_target(df_train, missing_col, target_col='stroke'):
    """
    Analyze relationship between missing values and target variable
    """
    # Patients WITH missing values
    with_missing = df_train[df_train[missing_col].isnull()]
    n_missing = len(with_missing)
    n_missing_positive = with_missing[target_col].sum()
    rate_missing = with_missing[target_col].mean()
    
    # Patients WITHOUT missing values
    without_missing = df_train[df_train[missing_col].notnull()]
    n_not_missing = len(without_missing)
    n_not_missing_positive = without_missing[target_col].sum()
    rate_not_missing = without_missing[target_col].mean()
    
    print(f"\n📊 {missing_col} Missingness vs {target_col} Analysis:")
    print(f"\nWhen {missing_col} IS missing:")
    print(f"- Total patients: {n_missing}")
    print(f"- Patients with {target_col}: {n_missing_positive}")
    print(f"- Rate: {n_missing_positive}/{n_missing} = {rate_missing:.4f} ({rate_missing*100:.2f}%)")
    
    print(f"\nWhen {missing_col} is NOT missing:")
    print(f"- Total patients: {n_not_missing}")
    print(f"- Patients with {target_col}: {n_not_missing_positive}")
    print(f"- Rate: {n_not_missing_positive}/{n_not_missing} = {rate_not_missing:.4f} ({rate_not_missing*100:.2f}%)")
    
    ratio = rate_missing / rate_not_missing if rate_not_missing > 0 else 0
    print(f"\n⚠️ Stroke rate is {ratio:.1f}x higher when {missing_col} is missing!")




def handle_outliers_iqr(df_train, columns, factor=1.5):
    """
    Detect outliers using IQR method
    
    Parameters:
    - df_train: Training dataframe
    - columns: List of columns to check for outliers
    - factor: IQR multiplier (default 1.5 for standard outliers)
    """
    outlier_info = []
    
    for col in columns:
        Q1 = df_train[col].quantile(0.25)
        Q3 = df_train[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = df_train[(df_train[col] < lower_bound) | (df_train[col] > upper_bound)]
        
        if len(outliers) > 0:
            outlier_info.append({
                'Column': col,
                'Lower_Bound': round(lower_bound, 2),
                'Upper_Bound': round(upper_bound, 2),
                'Outlier_Count': len(outliers),
                'Outlier_Percentage': round((len(outliers) / len(df_train)) * 100, 2),
                'Outlier_Stroke_Rate': round(outliers['stroke'].mean() * 100, 2)
            })
    
    if outlier_info:
        outlier_df = pd.DataFrame(outlier_info)
        print("🔍 Outlier Analysis (IQR Method):")
        display(outlier_df)
        
        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(columns):
            df_train.boxplot(column=col, by='stroke', ax=axes[idx])
            axes[idx].set_title(f'{col} by Stroke Status')
            axes[idx].set_xlabel('Stroke')
            axes[idx].set_ylabel(col)
        
        plt.suptitle('Outlier Detection using IQR Method', y=1.02)
        plt.tight_layout()
        plt.show()
        
        print("\n📊 Note: IQR method is more robust than Z-score for skewed distributions")
        print("Consider keeping outliers if they represent valid extreme cases")
    else:
        print("✅ No outliers detected using IQR method")



# =============================================================================
# 5. FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def create_domain_features(X_train, X_test):
    """
    Create domain-driven features based on medical knowledge
    
    Parameters:
    - X_train, X_test: Training and test feature sets
    
    Returns:
    - X_train_fe, X_test_fe: Feature sets with new features added
    """

    X_train_fe = X_train.copy()
    X_test_fe = X_test.copy()
    
    # 1. BMI Missing Indicator (important finding from preprocessing!)
    X_train_fe['bmi_missing'] = X_train_fe['bmi'].isnull().astype(int)
    X_test_fe['bmi_missing'] = X_test_fe['bmi'].isnull().astype(int)
    
    # 2. Age Groups
    age_bins = [0, 45, 65, 100]
    age_labels = ['young', 'middle_aged', 'senior']
    X_train_fe['age_group'] = pd.cut(X_train_fe['age'], bins=age_bins, labels=age_labels)
    X_test_fe['age_group'] = pd.cut(X_test_fe['age'], bins=age_bins, labels=age_labels)
    
    # 3. Is Senior (binary)
    X_train_fe['is_senior'] = (X_train_fe['age'] >= 65).astype(int)
    X_test_fe['is_senior'] = (X_test_fe['age'] >= 65).astype(int)
    
    # 4. BMI Categories (WHO standards)
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
    X_train_fe['bmi_category'] = pd.cut(X_train_fe['bmi'], bins=bmi_bins, labels=bmi_labels)
    X_test_fe['bmi_category'] = pd.cut(X_test_fe['bmi'], bins=bmi_bins, labels=bmi_labels)
    
    # 5. Glucose Level Categories
    glucose_bins = [0, 100, 125, 200, 300]
    glucose_labels = ['normal', 'prediabetic', 'diabetic', 'very_high']
    X_train_fe['glucose_category'] = pd.cut(X_train_fe['avg_glucose_level'], 
                                           bins=glucose_bins, labels=glucose_labels)
    X_test_fe['glucose_category'] = pd.cut(X_test_fe['avg_glucose_level'], 
                                          bins=glucose_bins, labels=glucose_labels)
    
    # 6. Multiple Risk Factors Count
    risk_factors = ['hypertension', 'heart_disease', 'is_senior']
    X_train_fe['risk_factor_count'] = X_train_fe[risk_factors].sum(axis=1)
    X_test_fe['risk_factor_count'] = X_test_fe[risk_factors].sum(axis=1)
    
    # 7. High Risk Group (multiple conditions)
    X_train_fe['high_risk_group'] = (
        (X_train_fe['risk_factor_count'] >= 2) | 
        (X_train_fe['glucose_category'] == 'diabetic') |
        (X_train_fe['bmi_category'] == 'obese')
    ).astype(int)
    
    X_test_fe['high_risk_group'] = (
        (X_test_fe['risk_factor_count'] >= 2) | 
        (X_test_fe['glucose_category'] == 'diabetic') |
        (X_test_fe['bmi_category'] == 'obese')
    ).astype(int)
    
    # 8. BMI-Glucose Interaction (if not missing)
    X_train_fe['bmi_glucose_ratio'] = X_train_fe['bmi'] / X_train_fe['avg_glucose_level']
    X_test_fe['bmi_glucose_ratio'] = X_test_fe['bmi'] / X_test_fe['avg_glucose_level']
    
    print("✅ Created 8 new features:")
    new_features = ['bmi_missing', 'age_group', 'is_senior', 'bmi_category', 
                   'glucose_category', 'risk_factor_count', 'high_risk_group', 
                   'bmi_glucose_ratio']
    
    for feat in new_features:
        print(f"  - {feat}")
    
    print(f"\nOriginal features: {X_train.shape[1]}")
    print(f"New total features: {X_train_fe.shape[1]}")
    
    return X_train_fe, X_test_fe


# =============================================================================
# 6. COMPREHENSIVE EDA FUNCTIONS
# =============================================================================


def plot_all_distributions(df_train, target='stroke'):
    """
    Plot distributions for all features (numerical and categorical)
    """
    numerical_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target in numerical_features:
        numerical_features.remove(target)
    
    print("📊 Numerical Feature Distributions\n")
    
    n_numerical = len(numerical_features)
    n_cols = 3
    n_rows = (n_numerical + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(numerical_features):
        df_plot = df_train[[col, target]].dropna()
        
        ax = axes[idx]
        sns.boxplot(data=df_plot, x=target, y=col, ax=ax, palette=['lightblue', 'lightcoral'])
        
        ax.set_title(f'{col} Distribution by Stroke Status')
        ax.set_xlabel('Stroke')
        
        medians = df_plot.groupby(target)[col].median()
        for i, median in enumerate(medians):
            ax.text(i, median, f'{median:.1f}', ha='center', va='bottom', fontweight='bold')
    
    for idx in range(len(numerical_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Categorical Feature Distributions\n")
    
    n_categorical = len(categorical_features)
    n_rows = (n_categorical + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_categorical == 1 else axes
    
    for idx, col in enumerate(categorical_features):
        ax = axes[idx]
        
        cross_tab = pd.crosstab(df_train[col], df_train[target], normalize='index') * 100
        cross_tab.plot(kind='bar', stacked=True, ax=ax, color=['lightblue', 'lightcoral'])
        
        ax.set_title(f'{col} vs Stroke Rate')
        ax.set_xlabel(col)
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Stroke', labels=['No Stroke', 'Stroke'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        stroke_rates = df_train.groupby(col)[target].mean() * 100
        for i, (cat, rate) in enumerate(stroke_rates.items()):
            ax.text(i, 50, f'{rate:.1f}%', ha='center', va='center', fontweight='bold')
    
    for idx in range(len(categorical_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()




def check_multicollinearity(df_train):
    """
    Check multicollinearity using VIF for numerical features
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    numerical_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
    if 'stroke' in numerical_features:
        numerical_features.remove('stroke')
    
    df_vif = df_train[numerical_features].dropna()
    
    vif_data = []
    for i in range(len(numerical_features)):
        vif_value = variance_inflation_factor(df_vif.values, i)
        vif_data.append({
            'Feature': numerical_features[i],
            'VIF': vif_value,
            'Multicollinearity': 'High' if vif_value > 10 else 'Moderate' if vif_value > 5 else 'Low'
        })
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    
    print("📊 Variance Inflation Factor (VIF) Analysis\n")
    display(vif_df)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(vif_df)), vif_df['VIF'], 
                    color=['red' if x > 10 else 'orange' if x > 5 else 'green' for x in vif_df['VIF']])
    
    plt.axhline(y=5, color='orange', linestyle='--', label='Moderate (VIF=5)')
    plt.axhline(y=10, color='red', linestyle='--', label='High (VIF=10)')
    
    plt.xlabel('Features')
    plt.ylabel('VIF Score')
    plt.title('Multicollinearity Check - Variance Inflation Factor')
    plt.xticks(range(len(vif_df)), vif_df['Feature'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for i, (idx, row) in enumerate(vif_df.iterrows()):
        plt.text(i, row['VIF'] + 0.5, f'{row["VIF"]:.1f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        print("\n⚠️ High multicollinearity detected in:", list(high_vif['Feature']))
        print("Consider removing or combining these features")
    else:
        print("\n✅ No severe multicollinearity detected (all VIF < 10)")




def analyze_feature_correlations(df_train, target='stroke'):
    """
    Analyze correlations using appropriate methods for different feature types
    """

    numerical_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target in numerical_features:
        numerical_features.remove(target)
    
    # 1. Numerical correlations (Pearson + Spearman)
    print("📊 Numerical Feature Correlations\n")
    
    # Pearson correlation (linear relationships)
    pearson_corr = df_train[numerical_features + [target]].corr(method='pearson')[target].drop(target).sort_values(ascending=False)
    
    # Spearman correlation (monotonic relationships)
    spearman_corr = df_train[numerical_features + [target]].corr(method='spearman')[target].drop(target).sort_values(ascending=False)
    
    corr_df = pd.DataFrame({
        'Feature': pearson_corr.index,
        'Pearson_r': pearson_corr.values,
        'Spearman_rho': spearman_corr.values,
        'Abs_Diff': abs(pearson_corr.values - spearman_corr.values)
    })
    
    print("Numerical Features vs Stroke (sorted by Pearson correlation):")
    display(corr_df.round(3))
    
    nonlinear = corr_df[corr_df['Abs_Diff'] > 0.1]
    if len(nonlinear) > 0:
        print("\n⚠️ Potential non-linear relationships (Pearson ≠ Spearman):")
        display(nonlinear)
    
    # 2. Categorical associations (Chi-square test)
    print("\n📊 Categorical Feature Associations (Chi-square test)\n")
    
    from scipy.stats import chi2_contingency
    
    chi_results = []
    for cat_feature in categorical_features:
        contingency = pd.crosstab(df_train[cat_feature], df_train[target])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0]-1, contingency.shape[1]-1)
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        chi_results.append({
            'Feature': cat_feature,
            'Chi2': chi2,
            'p_value': p_value,
            'Cramers_V': cramers_v,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    chi_df = pd.DataFrame(chi_results).sort_values('Cramers_V', ascending=False)
    display(chi_df.round(4))
    
    create_correlation_heatmaps(df_train)
    
    return corr_df, chi_df




def create_correlation_heatmaps(df_train):
    """
    Create two heatmaps: Pearson and Spearman correlations for numerical features
    """

    numerical_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
    
    pearson_corr = df_train[numerical_features].corr(method='pearson')
    spearman_corr = df_train[numerical_features].corr(method='spearman')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Pearson correlation heatmap
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
    sns.heatmap(pearson_corr, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax1,
                cbar_kws={"shrink": .8})
    ax1.set_title('Pearson Correlation (Linear)', fontsize=14)
    
    # Spearman correlation heatmap
    sns.heatmap(spearman_corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax2,
                cbar_kws={"shrink": .8})
    ax2.set_title('Spearman Correlation (Monotonic)', fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    diff_matrix = abs(pearson_corr - spearman_corr)
    non_linear_pairs = []
    
    for i in range(len(diff_matrix.columns)):
        for j in range(i+1, len(diff_matrix.columns)):
            if diff_matrix.iloc[i, j] > 0.1:
                non_linear_pairs.append({
                    'Feature1': diff_matrix.columns[i],
                    'Feature2': diff_matrix.columns[j],
                    'Pearson': pearson_corr.iloc[i, j],
                    'Spearman': spearman_corr.iloc[i, j],
                    'Difference': diff_matrix.iloc[i, j]
                })
    
    if non_linear_pairs:
        print("\n⚠️ Potential non-linear relationships (|Pearson - Spearman| > 0.1):")
        display(pd.DataFrame(non_linear_pairs).round(3))



        
def perform_statistical_inference(df_train):
    """
    Perform comprehensive statistical hypothesis testing with confidence intervals
    """
    from scipy import stats
    from statsmodels.stats.proportion import proportions_ztest

    stroke_group = df_train[df_train['stroke'] == 1]
    no_stroke_group = df_train[df_train['stroke'] == 0]
    
    print("🎯 TARGET POPULATION: Adults (18+) from healthcare settings at risk for stroke")
    print("Based on electronic health records with demographic and clinical data\n")
    
    print("="*70)
    print("STATISTICAL HYPOTHESIS TESTING (α = 0.05)")
    print("="*70)
    
    # 1. AGE ANALYSIS
    print("\n📊 TEST 1: AGE DIFFERENCE BETWEEN GROUPS")
    print("-"*50)
    print("H₀: μ_stroke = μ_no_stroke (mean age is equal)")
    print("H₁: μ_stroke ≠ μ_no_stroke (mean age differs)")
    
    age_stroke = stroke_group['age'].dropna()
    age_no_stroke = no_stroke_group['age'].dropna()
    
    t_age, p_age = stats.ttest_ind(age_stroke, age_no_stroke)
    
    ci_stroke = stats.t.interval(0.95, len(age_stroke)-1, 
                                 loc=age_stroke.mean(), 
                                 scale=stats.sem(age_stroke))
    ci_no_stroke = stats.t.interval(0.95, len(age_no_stroke)-1, 
                                    loc=age_no_stroke.mean(), 
                                    scale=stats.sem(age_no_stroke))
    
    print(f"\nStroke group: n={len(age_stroke)}, mean={age_stroke.mean():.1f} years")
    print(f"95% CI: [{ci_stroke[0]:.1f}, {ci_stroke[1]:.1f}]")
    print(f"\nNo-stroke group: n={len(age_no_stroke)}, mean={age_no_stroke.mean():.1f} years")
    print(f"95% CI: [{ci_no_stroke[0]:.1f}, {ci_no_stroke[1]:.1f}]")
    print(f"\nt-statistic: {t_age:.3f}, p-value: {p_age:.3e}")
    print(f"Decision: {'REJECT H₀' if p_age < 0.05 else 'FAIL TO REJECT H₀'}")
    print(f"Interpretation: Stroke patients are {'significantly' if p_age < 0.05 else 'not significantly'} older")
    
    # 2. GLUCOSE ANALYSIS
    print("\n\n📊 TEST 2: GLUCOSE LEVELS")
    print("-"*50)
    print("Test: Independent t-test (one-tailed)")
    print("H₀: μ_stroke ≤ μ_no_stroke  (stroke group has same or lower mean glucose)")
    print("H₁: μ_stroke > μ_no_stroke   (stroke group has higher mean glucose)")

    glucose_stroke = stroke_group['avg_glucose_level'].dropna()
    glucose_no_stroke = no_stroke_group['avg_glucose_level'].dropna()

    n_stroke = len(glucose_stroke)
    n_no_stroke = len(glucose_no_stroke)
    mean_stroke = glucose_stroke.mean()
    mean_no_stroke = glucose_no_stroke.mean()
    diff = mean_stroke - mean_no_stroke

    # Two-tailed t-test first
    t_stat_glucose, p_val_two_tailed = stats.ttest_ind(glucose_stroke, glucose_no_stroke, equal_var=False)

    # One-tailed p-value
    p_val_one_tailed = p_val_two_tailed / 2 if t_stat_glucose > 0 else 1 - (p_val_two_tailed / 2)

    # Cohen's d
    pooled_std = np.sqrt(((n_stroke - 1)*glucose_stroke.std()**2 + (n_no_stroke - 1)*glucose_no_stroke.std()**2) / 
                         (n_stroke + n_no_stroke - 2))
    cohen_d = diff / pooled_std
    effect_size_label = (
        "small" if abs(cohen_d) < 0.5 else 
        "medium" if abs(cohen_d) < 0.8 else 
        "large"
    )

    print(f"\nStroke group: n={n_stroke}, mean={mean_stroke:.1f} mg/dL")
    print(f"No-stroke group: n={n_no_stroke}, mean={mean_no_stroke:.1f} mg/dL")
    print(f"Difference: {diff:.1f} mg/dL")
    print(f"\nt-statistic: {t_stat_glucose:.3f}")
    print(f"p-value (one-tailed): {p_val_one_tailed:.3e}")
    print(f"Effect size (Cohen's d): {cohen_d:.3f} → {effect_size_label.capitalize()} effect")
    print(f"Decision: {'✅ REJECT H₀ (Significant difference)' if p_val_one_tailed < 0.05 else '❌ FAIL TO REJECT H₀ (Not significant)'}")

    
    # 3. HYPERTENSION ANALYSIS
    print("\n\n📊 TEST 3: HYPERTENSION PREVALENCE")
    print("-"*50)
    print("H₀: p_stroke = p_no_stroke (equal proportions)")
    print("H₁: p_stroke ≠ p_no_stroke (different proportions)")
    
    hyper_stroke = (stroke_group['hypertension'] == 1).sum()
    hyper_no_stroke = (no_stroke_group['hypertension'] == 1).sum()
    n_stroke = len(stroke_group)
    n_no_stroke = len(no_stroke_group)
    
    p_stroke = hyper_stroke / n_stroke
    p_no_stroke = hyper_no_stroke / n_no_stroke
    
    from statsmodels.stats.proportion import proportion_confint
    ci_p_stroke = proportion_confint(hyper_stroke, n_stroke, method='wilson')
    ci_p_no_stroke = proportion_confint(hyper_no_stroke, n_no_stroke, method='wilson')
    
    count = np.array([hyper_stroke, hyper_no_stroke])
    nobs = np.array([n_stroke, n_no_stroke])
    z_stat, p_prop = proportions_ztest(count, nobs)
    
    print(f"\nStroke group: {hyper_stroke}/{n_stroke} = {p_stroke:.1%}")
    print(f"95% CI: [{ci_p_stroke[0]:.1%}, {ci_p_stroke[1]:.1%}]")
    print(f"\nNo-stroke group: {hyper_no_stroke}/{n_no_stroke} = {p_no_stroke:.1%}")
    print(f"95% CI: [{ci_p_no_stroke[0]:.1%}, {ci_p_no_stroke[1]:.1%}]")
    print(f"\nz-statistic: {z_stat:.3f}, p-value: {p_prop:.3e}")
    print(f"Decision: {'REJECT H₀' if p_prop < 0.05 else 'FAIL TO REJECT H₀'}")
    
    print("\n\n" + "="*70)
    print("🔍 CONNECTION TO CORRELATION ANALYSIS")
    print("="*70)
    print("\nThe hypothesis tests confirm and extend the correlation findings:")
    
    print(f"• Age showed moderate correlation (r=0.24) with stroke, and t-test confirms")
    print(f"  stroke patients are {age_stroke.mean()-age_no_stroke.mean():.0f} years older on average")
    print(f"• Despite lower correlation (r=0.13), glucose shows clinically meaningful")
    print(f"  difference of {glucose_stroke.mean()-glucose_no_stroke.mean():.0f} mg/dL between groups")
    print(f"• Hypertension prevalence is {p_stroke/p_no_stroke:.1f}x higher in stroke patients")
    print("\nThese findings validate the feature engineering strategy of creating")
    print("age-based features (is_senior, age_group) and risk factor combinations.")


    
# =============================================================================
# 7. PREPROCESSING PIPELINE FUNCTIONS
# =============================================================================

def create_preprocessing_pipeline(continuous_features, categorical_features, binary_features):
    """
    Create preprocessing pipeline for continuous, categorical, and binary features
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    
    # Preprocessing for continuous features
    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # Combine all preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_transformer, continuous_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # This keeps binary features as-is
    )
    
    return preprocessor



# =============================================================================
# 8. RESAMPLING FUNCTIONS
# =============================================================================


def compare_resampling_methods(X_train_processed, y_train, X_test_processed, y_test):
    """
    Compare different resampling strategies
    Note: Expects already preprocessed data (no missing values, already scaled)
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    results = {}
    
    # 1. No resampling (baseline)
    lr_base = LogisticRegression(random_state=42, max_iter=1000)
    lr_base.fit(X_train_processed, y_train)
    y_pred_base = lr_base.predict(X_test_processed)
    
    results['No Resampling'] = {
        'precision': precision_score(y_test, y_pred_base),
        'recall': recall_score(y_test, y_pred_base),
        'f1': f1_score(y_test, y_pred_base)
    }
    
    # 2. Random Oversampling
    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X_train_processed, y_train)
    
    lr_ros = LogisticRegression(random_state=42, max_iter=1000)
    lr_ros.fit(X_ros, y_ros)
    y_pred_ros = lr_ros.predict(X_test_processed)
    
    results['Random Oversampling'] = {
        'precision': precision_score(y_test, y_pred_ros),
        'recall': recall_score(y_test, y_pred_ros),
        'f1': f1_score(y_test, y_pred_ros)
    }
    
    # 3. SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train_processed, y_train)
    
    lr_smote = LogisticRegression(random_state=42, max_iter=1000)
    lr_smote.fit(X_smote, y_smote)
    y_pred_smote = lr_smote.predict(X_test_processed)
    
    results['SMOTE'] = {
        'precision': precision_score(y_test, y_pred_smote),
        'recall': recall_score(y_test, y_pred_smote),
        'f1': f1_score(y_test, y_pred_smote)
    }
    
    results_df = pd.DataFrame(results).T
    print("📊 Resampling Methods Comparison:")
    display(results_df.round(3))
    
    print("\n📈 Class Distribution After Resampling:")
    print(f"Original: {np.bincount(y_train)}")
    print(f"Random Oversampling: {np.bincount(y_ros)}")
    print(f"SMOTE: {np.bincount(y_smote)}")
    
    return results_df



# =============================================================================
# 9. MODEL DEVELOPMENT & TUNING FUNCTIONS
# =============================================================================

def compare_models_baseline(X_train, y_train, X_test, y_test, models_dict):
    """
    Compare multiple models with default parameters
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import time
    
    results = []
    fitted_models = {}
    
    for name, model in models_dict.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        fitted_models[name] = model  # ADD THIS LINE
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        
        train_time = time.time() - start_time
        
        positive_preds = sum(y_pred)
        print(f"✅ {name} - F1: {f1:.3f}, Recall: {recall:.3f}, Positive predictions: {positive_preds}/{len(y_pred)}")
        
        results.append({
            'Model': name,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC-AUC': roc_auc,
            'Train_Time': train_time
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1', ascending=False)
    
    print("\n📊 Model Comparison Results:")
    display(results_df.round(3))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(results_df))
    width = 0.25
    
    color_f1 = '#1f77b4'
    color_recall = '#6baed6' 
    
    axes[0].bar(x - width/2, results_df['F1'], width, label='F1', color='#2c5aa0', alpha=0.8)
    axes[0].bar(x + width/2, results_df['Recall'], width, label='Recall', color='#6baed6', alpha=0.8)
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Score')
    axes[0].set_title('F1 vs Recall by Model')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(0.1, results_df[['F1', 'Recall']].max().max() * 1.2))
    
    axes[1].bar(x, results_df['ROC-AUC'], width=0.4, color='#2c5aa0', alpha=0.8)  # Slimmer width
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_title('ROC-AUC by Model')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    
    return results_df, fitted_models



    
def compare_models_with_resampling(X_train_processed, y_train, X_test_processed, y_test, models_dict):
    """
    Compare models using SMOTE resampled data
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    import time
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    print(f"✅ SMOTE applied: {len(y_train)} → {len(y_train_resampled)} samples")
    print(f"Class distribution: {np.bincount(y_train_resampled)}")
    
    results = []
    fitted_models = {} 
    
    for name, model in models_dict.items():
        print(f"\nTraining {name} on resampled data...")
        start_time = time.time()
        
        # Train model on resampled data
        model.fit(X_train_resampled, y_train_resampled)
        fitted_models[name] = model  # ADD THIS LINE
        
        # Predictions on original test set
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        
        train_time = time.time() - start_time
        
        results.append({
            'Model': name,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC-AUC': roc_auc,
            'Train_Time': train_time
        })
        
        print(f"✅ {name} - F1: {f1:.3f}, Recall: {recall:.3f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1', ascending=False)
    
    print("\n📊 Model Comparison Results (with SMOTE):")
    display(results_df.round(3))
    
    return results_df, fitted_models



# =============================================================================
# 10. FEATURE IMPORTANCE AND SELECTION
# =============================================================================


def plot_feature_importance(model, feature_names, model_name="", top_n=10, model_type="tree", title=None):
    """
    Plot feature importance or coefficients from a model.
    Parameters:
    - model: trained model (must have .coef_ or .feature_importances_)
    - feature_names: list of feature names after preprocessing
    - model_name: name of the model (e.g., "Logistic Regression")
    - top_n: number of top features to display (default 10)
    - model_type: "linear", "tree", or "permutation"
    - title: optional plot title
    """
    if model_type == "linear":
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0]) 
        else:
            raise ValueError("Model does not have .coef_ attribute.")
        label = "Coefficient (abs)"
    elif model_type == "tree":
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            raise ValueError("Model does not have .feature_importances_ attribute.")
        label = "Importance"
    elif model_type == "permutation":
        if hasattr(model, "importances_mean"):
            importances = model.importances_mean
        else:
            raise ValueError("Permutation importance object must have .importances_mean")
        label = "Importance (permuted)"
    else:
        raise ValueError("Unsupported model_type. Use 'linear', 'tree', or 'permutation'.")
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color="#1976D2")
    plt.xlabel(label)
    
    if title:
        plt.title(title)
    else:
        plt.title(f"Top {top_n} Important Features - {model_name}" if model_name else f"Top {top_n} Important Features")
    
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return importance_df



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
from IPython.display import display

def compare_feature_sets(X_train_full, X_test_full, X_train_top, X_test_top,
                         y_train, y_test, feature_names_all, feature_names_top,
                         model=None, top_n_label="Top Features", all_label="All Features"):
    """
    Compare model performance using all features vs selected top features (after preprocessing).
    """
    
    # Step 1: Choose model
    if model is None:
        model_full = LogisticRegression(random_state=42, max_iter=1000)
        model_top = LogisticRegression(random_state=42, max_iter=1000)
    else:
        model_full = model
        model_top = model

    # Step 2: Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_full_sm, y_train_full_sm = smote.fit_resample(X_train_full, y_train)
    X_train_top_sm, y_train_top_sm = smote.fit_resample(X_train_top, y_train)

    # Step 3: Train both models
    model_full.fit(X_train_full_sm, y_train_full_sm)
    model_top.fit(X_train_top_sm, y_train_top_sm)

    # Step 4: Predict on original test set
    y_pred_full = model_full.predict(X_test_full)
    y_pred_top = model_top.predict(X_test_top)

    metrics = []

    for name, y_pred in zip([all_label, top_n_label], [y_pred_full, y_pred_top]):
        metrics.append({
            "Feature Set": name,
            "F1 Score": round(f1_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Accuracy": round(accuracy_score(y_test, y_pred), 3)
        })

    summary_df = pd.DataFrame(metrics)

    print("\n📊 Classification Report – All Features:")
    print(classification_report(y_test, y_pred_full, target_names=["No Stroke", "Stroke"]))

    print("\n📊 Classification Report – Top Features:")
    print(classification_report(y_test, y_pred_top, target_names=["No Stroke", "Stroke"]))

    print("\n📈 Summary Comparison:")
    display(summary_df)

    return summary_df





def plot_confusion_matrices(models_dict, X_test, y_test, title="Model Comparison"):
    """
    Plot confusion matrices for multiple models in a grid
    
    Parameters:
    - models_dict: Dictionary of fitted models {'name': model}
    - X_test: Test features
    - y_test: Test labels
    - title: Overall plot title
    """
    from sklearn.metrics import confusion_matrix, recall_score, precision_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    n_models = len(models_dict)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    axes = axes.ravel() if n_models > 1 else [axes]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        

        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Stroke', 'Stroke'],
                    yticklabels=['No Stroke', 'Stroke'],
                    ax=axes[idx], cbar=False)
        
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        
        axes[idx].set_title(f'{name}\nRecall: {recall:.1%} | Precision: {precision:.1%}', 
                           fontsize=12, pad=10)
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    for idx in range(len(models_dict), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()



# =============================================================================
# 11. MODEL DEVELOPMENT, TUNING & FINAL EVALUATION
# =============================================================================


def tune_logistic_regression(X_train, y_train, apply_smote=True, scoring_metric='recall'):
    """
    Efficient hyperparameter tuning for Logistic Regression using GridSearchCV.
    """
    import warnings
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
    from sklearn.metrics import classification_report
    from sklearn.exceptions import ConvergenceWarning
    from imblearn.over_sampling import SMOTE
    from IPython.display import display

    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"✅ SMOTE applied: {len(y_train)} → {len(y_train_balanced)} samples")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],  
        'class_weight': [None, 'balanced']
    }

    total_combinations = len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['solver']) * len(param_grid['class_weight'])
    print(f"🔍 Testing {total_combinations} combinations...")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=10000),
        param_grid,
        cv=cv,
        scoring=['recall', 'precision', 'f1'],
        refit=scoring_metric,
        n_jobs=1, 
        verbose=0
    )

    grid_search.fit(X_train_balanced, y_train_balanced)

    best_model = grid_search.best_estimator_
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"📈 Best CV {scoring_metric}: {grid_search.best_score_:.3f}")

    # Top 5 models summary
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_models = results_df.nlargest(5, f'mean_test_{scoring_metric}')[
        ['params', f'mean_test_{scoring_metric}', 'mean_test_precision', 'mean_test_f1']
    ].round(3)
    
    pd.set_option('display.max_colwidth', None)
    print("\n🏆 Top 5 Models:")
    display(top_models)

    return best_model, results_df





def compare_model_progression(X_train_30, X_train_7, y_train, best_tuned_model):
    """
    Compare model progression using cross-validation on TRAINING DATA ONLY
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    import pandas as pd
    import numpy as np
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42)
    
    # 1. Baseline (30 features, no SMOTE)
    lr_base = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(lr_base, X_train_30, y_train, cv=cv, 
                           scoring='accuracy')
    results['Baseline LR (30 feat)'] = {
        'Accuracy': scores.mean(),
        'Precision': cross_val_score(lr_base, X_train_30, y_train, cv=cv, scoring='precision').mean(),
        'Recall': cross_val_score(lr_base, X_train_30, y_train, cv=cv, scoring='recall').mean(),
        'F1': cross_val_score(lr_base, X_train_30, y_train, cv=cv, scoring='f1').mean(),
        'ROC-AUC': cross_val_score(lr_base, X_train_30, y_train, cv=cv, scoring='roc_auc').mean()
    }
    
    # 2. LR + SMOTE (30 features)
    pipeline_30 = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    scores = cross_val_score(pipeline_30, X_train_30, y_train, cv=cv, scoring='accuracy')
    results['LR + SMOTE (30 feat)'] = {
        'Accuracy': scores.mean(),
        'Precision': cross_val_score(pipeline_30, X_train_30, y_train, cv=cv, scoring='precision').mean(),
        'Recall': cross_val_score(pipeline_30, X_train_30, y_train, cv=cv, scoring='recall').mean(),
        'F1': cross_val_score(pipeline_30, X_train_30, y_train, cv=cv, scoring='f1').mean(),
        'ROC-AUC': cross_val_score(pipeline_30, X_train_30, y_train, cv=cv, scoring='roc_auc').mean()
    }
    
    # 3. LR + SMOTE (7 features)
    pipeline_7 = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    scores = cross_val_score(pipeline_7, X_train_7, y_train, cv=cv, scoring='accuracy')
    results['LR + SMOTE (7 feat)'] = {
        'Accuracy': scores.mean(),
        'Precision': cross_val_score(pipeline_7, X_train_7, y_train, cv=cv, scoring='precision').mean(),
        'Recall': cross_val_score(pipeline_7, X_train_7, y_train, cv=cv, scoring='recall').mean(),
        'F1': cross_val_score(pipeline_7, X_train_7, y_train, cv=cv, scoring='f1').mean(),
        'ROC-AUC': cross_val_score(pipeline_7, X_train_7, y_train, cv=cv, scoring='roc_auc').mean()
    }
    
    # 4. Tuned model (7 features) - need to apply SMOTE first
    X_train_7_smote, y_train_smote = smote.fit_resample(X_train_7, y_train)
    scores = cross_val_score(best_tuned_model, X_train_7_smote, y_train_smote, cv=cv, scoring='accuracy')
    results['LR + SMOTE + Tuned (7 feat)'] = {
        'Accuracy': scores.mean(),
        'Precision': cross_val_score(best_tuned_model, X_train_7_smote, y_train_smote, cv=cv, scoring='precision').mean(),
        'Recall': cross_val_score(best_tuned_model, X_train_7_smote, y_train_smote, cv=cv, scoring='recall').mean(),
        'F1': cross_val_score(best_tuned_model, X_train_7_smote, y_train_smote, cv=cv, scoring='f1').mean(),
        'ROC-AUC': cross_val_score(best_tuned_model, X_train_7_smote, y_train_smote, cv=cv, scoring='roc_auc').mean()
    }
    
    results_df = pd.DataFrame(results).T.round(3)
    print("📊 Model Progression - Cross-Validation Results (Training Data Only):")
    display(results_df)
    
    print("\n⚠️ Note: All metrics are 5-fold cross-validation scores on training data.")
    print("Test set has NOT been touched!")
    
    return results_df




def final_model_evaluation(X_train, y_train, X_test, y_test, best_model_params, feature_names):
    """
    Final evaluation of the best model on hold-out test set
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
    from sklearn.metrics import precision_score, recall_score, f1_score
    from imblearn.over_sampling import SMOTE
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("⚠️ USING TEST DATA FOR THE FIRST TIME!")
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    final_model = LogisticRegression(**best_model_params)
    final_model.fit(X_train_smote, y_train_smote)
    
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    
    print("\n📊 FINAL MODEL PERFORMANCE:")
    print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    plt.title('Final Model - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    plot_performance_curves(y_test, y_proba)
    
    print("\n🎯 SUMMARY:")
    print(f"Features used: {len(feature_names)} ({', '.join(feature_names[:3])}...)")
    print(f"Recall: {recall_score(y_test, y_pred):.1%}")
    print(f"Precision: {precision_score(y_test, y_pred):.1%}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    
    return final_model, y_pred, y_proba



def plot_performance_curves(y_true, y_proba):
    """
    Plot ROC and PR curves
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()




def plot_confusion_matrices_custom(predictions_dict, y_true, title="Model Comparison"):
    """
    Plot confusion matrices for multiple models from predictions
    """
    from sklearn.metrics import confusion_matrix, recall_score, precision_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    n_models = len(predictions_dict)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel() if n_models > 1 else [axes]
    
    for idx, (name, y_pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Stroke', 'Stroke'],
                    yticklabels=['No Stroke', 'Stroke'],
                    ax=axes[idx], cbar=False)
        
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        
        axes[idx].set_title(f'{name}\nRecall: {recall:.1%} | Precision: {precision:.1%}', 
                           fontsize=11, pad=10)
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    for idx in range(len(predictions_dict), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 13. SAVING FINAL ARTIFACTS
# =============================================================================


def save_final_artifacts(model, pipeline, feature_names, metrics_dict, folder="artifacts"):
    """
    Saves model, pipeline, feature names, and evaluation metrics.
    """
    import os
    import json
    import joblib
    import pandas as pd

    os.makedirs(folder, exist_ok=True)

    joblib.dump(model, f"{folder}/final_model.pkl")
    joblib.dump(pipeline, f"{folder}/preprocessing_pipeline.pkl")

    with open(f"{folder}/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(f"{folder}/final_evaluation_report.csv", index=False)

    print(f"✅ Artifacts saved to folder: '{folder}'")




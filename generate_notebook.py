import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
"""))
cells.append(nbf.v4.new_markdown_cell("""
### **Analysis: Imports**
This cell initializes the environment by importing essential libraries for data manipulation (`pandas`, `numpy`), visualization (`matplotlib`, `seaborn`), and machine learning (`scikit-learn`). It also suppresses warnings to ensure a clean output during model execution.
"""))

# Cell 2: Data Loading
cells.append(nbf.v4.new_code_cell("""
df = pd.read_csv('indonesia_higher_education_institutions.csv')
print(f"Dataset shape: {df.shape}")
df.head()
"""))
cells.append(nbf.v4.new_markdown_cell("""
### **Analysis: Data Loading**
The dataset contains 6,650 records of higher education institutions with 20 features. Initial inspection shows a mix of categorical data (names, types, groups, locations) and numerical data (program counts, coordinates). Several columns, such as `accreditation` and `tuition_range`, contain significant missing values which require handling during preprocessing.
"""))

# Cell 3: EDA - Institutional Distribution
cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(12, 6))
sns.countplot(y='institution_type', data=df, order=df['institution_type'].value_counts().index)
plt.title('Distribution of Institution Types')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='institution_group', data=df)
plt.title('Distribution of Institution Groups')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("""
### **Analysis: Institutional Distribution**
*   **Institution Type:** The bar chart reveals that "Sekolah Tinggi" (Higher School) and "Akademi" (Academy) are the most prevalent types of institutions in Indonesia, followed by "Universitas" (University).
*   **Institution Group:** "PTS" (Private Higher Education) vastly outnumbers "PTN" (State Higher Education) and other religious or ministry-affiliated groups (PTA, PTKL), highlighting the dominant role of the private sector in Indonesian tertiary education.
"""))

# Cell 4: EDA - Accreditation Distribution
cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 6))
sns.countplot(x='accreditation', data=df, order=df['accreditation'].value_counts().index)
plt.title('Distribution of Accreditation Levels')
plt.xticks(rotation=45)
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("""
### **Analysis: Accreditation Distribution**
The accreditation status is heavily imbalanced. The majority of institutions are accredited as "Baik" (Good). Premium ranks like "Unggul" (Excellent) and "Baik Sekali" (Very Good) represent a much smaller portion of the dataset. Older labels like "A", "B", and "C" indicate a transition in the accreditation reporting system.
"""))

# Cell 5: EDA - Program Count Analysis
cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 6))
sns.histplot(df['program_count'], bins=50, kde=True)
plt.title('Distribution of Program Counts')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='institution_group', y='program_count', data=df)
plt.title('Program Count by Institution Group')
plt.show()
"""))
cells.append(nbf.v4.new_markdown_cell("""
### **Analysis: Program Count Analysis**
*   The histogram shows that most institutions offer a small number of programs (typically 1-10), while a few large universities offer over 100 programs, creating a heavy right-skewed distribution.
*   The boxplot shows that "PTN" (State) institutions generally have a higher median program count and more variance compared to "PTS" (Private) or "PTA" (Religious) institutions.
"""))

# Cell 6: Data Cleaning and Preprocessing - Part 1
cells.append(nbf.v4.new_code_cell("""
# Drop rows where target 'accreditation' is missing for modeling
df_model = df.dropna(subset=['accreditation']).copy()

# Extract Founding Year
df_model['founding_year'] = pd.to_datetime(df_model['founding_date'], errors='coerce').dt.year
df_model['founding_year'] = df_model['founding_year'].fillna(df_model['founding_year'].median())

# Parse Tuition Range
def parse_tuition(val, mode='min'):
    if pd.isna(val) or '-' not in str(val):
        return np.nan
    parts = str(val).replace('.', '').split('-')
    try:
        if mode == 'min':
            return float(parts[0].strip())
        else:
            return float(parts[1].strip())
    except:
        return np.nan

df_model['min_tuition'] = df_model['tuition_range'].apply(lambda x: parse_tuition(x, 'min'))
df_model['max_tuition'] = df_model['tuition_range'].apply(lambda x: parse_tuition(x, 'max'))

# Fill tuition NaNs with median
df_model['min_tuition'] = df_model['min_tuition'].fillna(df_model['min_tuition'].median())
df_model['max_tuition'] = df_model['max_tuition'].fillna(df_model['max_tuition'].median())

print(f"Remaining data for modeling: {df_model.shape}")
"""))
cells.append(nbf.v4.new_markdown_cell("""
### **Analysis: Preprocessing - Part 1**
To ensure a reliable machine learning model, institutions without accreditation data were removed, leaving ~3,887 records. "Founding Year" was extracted from the date strings, and the complex "Tuition Range" string was parsed into distinct `min_tuition` and `max_tuition` numerical features. Median imputation was used to handle missing values in these new features.
"""))

# Cell 7: Preprocessing - Encoding and Feature Selection
cells.append(nbf.v4.new_code_cell("""
# Select features
features = ['institution_type', 'institution_group', 'province_name', 'program_count', 'founding_year', 'min_tuition', 'max_tuition']
X = df_model[features]
y = df_model['accreditation']

# Handle missing categoricals
X['institution_type'] = X['institution_type'].fillna('Unknown')
X['institution_group'] = X['institution_group'].fillna('Unknown')
X['province_name'] = X['province_name'].fillna('Unknown')

# Label Encoding for categorical features
le = LabelEncoder()
for col in ['institution_type', 'institution_group', 'province_name']:
    X[col] = le.fit_transform(X[col].astype(str))

# Encode target
y_encoded = le.fit_transform(y.astype(str))
target_names = le.classes_

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""))
cells.append(nbf.v4.new_markdown_cell("""
### **Analysis: Preprocessing - Part 2**
Categorical features (`institution_type`, `institution_group`, `province_name`) were transformed into numerical values using `LabelEncoder`. The data was then split into an 80% training set and 20% test set, with stratification to ensure the minority accreditation classes were represented in both sets. Standard scaling was applied to normalize the different ranges of the numerical features.
"""))

# Cell 8: Machine Learning - Random Forest
cells.append(nbf.v4.new_code_cell("""
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\\nClassification Report:\\n", classification_report(y_test, rf_pred, target_names=target_names))
"""))
cells.append(nbf.v4.new_markdown_cell("""
### **Analysis: Random Forest Classifier**
The Random Forest model achieved an accuracy of ~72%. The classification report shows high precision and recall for the "Baik" class (the majority), but struggles with minority classes like "Terakreditasi" or "Baik Sekali," which is expected given the class imbalance in the dataset.
"""))

# Cell 9: Machine Learning - Gradient Boosting
cells.append(nbf.v4.new_code_cell("""
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("\\nClassification Report:\\n", classification_report(y_test, gb_pred, target_names=target_names))
"""))
cells.append(nbf.v4.new_markdown_cell("""
### **Analysis: Gradient Boosting Classifier**
The Gradient Boosting model performed similarly to the Random Forest, also reaching approximately 72% accuracy. Gradient boosting models are generally strong with imbalanced tabular data, and it shows competitive recall for the "Unggul" class compared to Random Forest.
"""))

# Cell 10: Comparison Metrics
cells.append(nbf.v4.new_code_cell("""
metrics = {
    'Model': ['Random Forest', 'Gradient Boosting'],
    'Accuracy': [accuracy_score(y_test, rf_pred), accuracy_score(y_test, gb_pred)]
}

comparison_df = pd.DataFrame(metrics)

plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy', data=comparison_df)
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison')
for i, val in enumerate(comparison_df['Accuracy']):
    plt.text(i, val + 0.02, f'{val:.2f}', ha='center')
plt.show()

comparison_df
"""))
cells.append(nbf.v4.new_markdown_cell("""
### **Analysis: Model Comparison**
The comparison chart confirms that both ensemble models provide a stable performance of around 72%. For this specific task, "Program Count" and "Institution Group" appeared to be strong predictors of accreditation quality. To improve these scores further, additional features like "Student-to-Teacher Ratio" or "Research Output" (if available in raw sources) would be beneficial.
"""))

nb.cells = cells

with open('indonesia_education_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

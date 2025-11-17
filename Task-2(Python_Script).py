"""
   Task 2: 
   (Credit Risk Prediction) """

# -------------------------------------------------------------------

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# ----------------------- Configuration -----------------------
DATA_PATH = r"C:/Users/Abdullah Umer/Desktop/DevelopersHub Corporation Internship/TASK 2/Credit Risk Prediction DataSet.csv"
DATA_DIR = Path(DATA_PATH).parent
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Visualization style choices
sns.set(style="whitegrid")
plt.rcParams["figure.facecolor"] = "#9400D3"
plt.rcParams["axes.facecolor"] = "#ffffff"
plt.rcParams["axes.edgecolor"] = "#FFFFFF"
plt.rcParams["axes.labelcolor"] = "#FFFFFF"
plt.rcParams["xtick.color"] = "#F7DFDF"
plt.rcParams["ytick.color"] = "#F7DCDC"


# Use a friendly palette for plots
PALETTE = sns.color_palette("dark")
DIVERGING = sns.color_palette("RdYlBu", as_cmap=False)

warnings.filterwarnings("ignore")  



# ----------------------- Helper Functions -----------------------

def save_fig(fig, name: str):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def overview(df: pd.DataFrame):
    print("\n--- Dataset overview ---")
    print(df.info())
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nSample rows:\n", df.head())




# ----------------------- Preprocessing -----------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using sensible defaults:
    - Numerical: median
    - Categorical: mode
    - Loan_ID left as is (identifier)
    """
    df = df.copy()

    # Identify numeric and categorical cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Some numeric-like columns might be parsed as object
    # We'll explicitly convert known numeric columns if present
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
        if col in df.columns and df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=[object]).columns.tolist()

    # Fill numerics with median
    for col in numeric_cols:
        med = df[col].median()
        df[col] = df[col].fillna(med)

    # Fill categoricals with mode
    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])
            else:
                df[col] = df[col].fillna("Unknown")

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize Loan_Status target to 0/1
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    # Clean Dependents: replace '3+' with 3 and convert to numeric
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", "3")
        df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0).astype(int)

    # Create TotalIncome
    if all(col in df.columns for col in ["ApplicantIncome", "CoapplicantIncome"]):
        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Log transform skewed numeric features for better plots & modelling
    for col in ["LoanAmount", "ApplicantIncome", "CoapplicantIncome", "TotalIncome"]:
        if col in df.columns:
            # create a safe log column
            new_col = f"{col}_log"
            df[new_col] = np.log1p(df[col].astype(float))

    return df






# ----------------------- Visualizations (20 plots) -----------------------

def create_visualizations(df: pd.DataFrame):
    """Create and save 20 high-quality visualizations in the output folder."""
    print("Creating visualizations and saving to:", OUTPUT_DIR)

    # Ensure categorical mapping for plotting
    if "Loan_Status" in df.columns:
        df["Loan_Status_Label"] = df["Loan_Status"].map({1: "Approved/No Default", 0: "Not Approved/Default"})
    else:
        df["Loan_Status_Label"] = "Unknown"

    # 1. LoanAmount distribution (hist)
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(df["LoanAmount"].dropna(), kde=True, palette=PALETTE, color="purple")
    plt.title("Loan Amount Distribution")
    save_fig(fig, "01_loanamount_distribution")

    # 2. LoanAmount (log) distribution
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(df["LoanAmount_log"], kde=True, color="green")
    plt.title("Loan Amount (log) Distribution")
    save_fig(fig, "02_loanamount_log_distribution")

    # 3. ApplicantIncome distribution
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(df["ApplicantIncome"], bins=40)
    plt.title("Applicant Income Distribution")
    save_fig(fig, "03_applicantincome_distribution")

    # 4. ApplicantIncome (log)
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(df["ApplicantIncome_log"], bins=40, color="violet")
    plt.title("Applicant Income (log) Distribution")
    save_fig(fig, "04_applicantincome_log")

    # 5. TotalIncome (log)
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(df["TotalIncome_log"].dropna(), kde=True, color="red")
    plt.title("Total Income (log) Distribution")
    save_fig(fig, "05_totalincome_log")

    # 6. Countplot: Education vs Loan Status
    if "Education" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Education", hue="Loan_Status_Label", palette="Set2")
        plt.title("Education vs Loan Status")
        save_fig(fig, "06_education_vs_loanstatus")

    # 7. Countplot: Self_Employed vs Loan Status
    if "Self_Employed" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Self_Employed", hue="Loan_Status_Label", palette="Set1")
        plt.title("Self Employed vs Loan Status")
        save_fig(fig, "07_selfemployed_vs_loanstatus")

    # 8. Credit History vs Loan Status
    if "Credit_History" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="Credit_History", y="LoanAmount", ci=None, palette="Accent")
        plt.title("Credit History vs Average Loan Amount")
        save_fig(fig, "08_credithistory_vs_loanamount")

        fig = plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Credit_History", hue="Loan_Status_Label", palette="dark")
        plt.title("Credit History vs Loan Status")
        save_fig(fig, "09_credithistory_vs_loanstatus")

    # 10. Property Area vs Loan Status
    if "Property_Area" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Property_Area", hue="Loan_Status_Label", palette="tab10")
        plt.title("Property Area vs Loan Status")
        save_fig(fig, "10_propertyarea_vs_loanstatus")

    # 11. Boxplot: LoanAmount by Education
    if "Education" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Education", y="LoanAmount", palette="pastel")
        plt.title("Loan Amount by Education")
        save_fig(fig, "11_loanamount_by_education")

    # 12. Boxplot: TotalIncome by Loan Status
    if "TotalIncome" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Loan_Status_Label", y="TotalIncome", palette="muted")
        plt.title("Total Income by Loan Status")
        save_fig(fig, "12_totalincome_by_loanstatus")

    # 13. Scatter: TotalIncome vs LoanAmount
    if "TotalIncome" in df.columns:
        fig = plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="TotalIncome", y="LoanAmount", hue="Loan_Status_Label", palette="Dark2")
        plt.title("Total Income vs Loan Amount")
        save_fig(fig, "13_totalincome_vs_loanamount")

    # 14. Violin: ApplicantIncome_log by Loan Status
    fig = plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="Loan_Status_Label", y="ApplicantIncome_log", palette="Set3")
    plt.title("Applicant Income (log) by Loan Status")
    save_fig(fig, "14_applicantincome_log_violin_by_status")

    # 15. Heatmap: Correlation among numeric features
    fig = plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu")
    plt.title("Correlation Matrix (numeric features)")
    save_fig(fig, "15_correlation_heatmap")

    # 16. Countplot: Gender vs Loan Status
    if "Gender" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Gender", hue="Loan_Status_Label", palette="Set2")
        plt.title("Gender vs Loan Status")
        save_fig(fig, "16_gender_vs_loanstatus")

    # 17. Bar: Mean LoanAmount by Dependents
    if "Dependents" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        df_group = df.groupby("Dependents")["LoanAmount"].mean().reset_index()
        sns.barplot(data=df_group, x="Dependents", y="LoanAmount", palette="Greens_d")
        plt.title("Average Loan Amount by Dependents")
        save_fig(fig, "17_avgloanamount_by_dependents")

    # 18. KDE plots of LoanAmount_log by Loan Status
    fig = plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x="LoanAmount_log", hue="Loan_Status_Label", fill=True)
    plt.title("KDE of Loan Amount (log) by Loan Status")
    save_fig(fig, "18_kde_loanamount_log_by_status")

    # 19. Pairplot of selected numeric log features
    pair_cols = [c for c in df.columns if c.endswith("_log")][:4]
    if len(pair_cols) >= 2:
        # Pairplot is large; save separately
        pp = sns.pairplot(df[pair_cols + ["Loan_Status_Label"]], hue="Loan_Status_Label", diag_kind="kde", corner=True)
        pp.fig.suptitle("Pairplot of log-features (subset)", y=1.02)
        pp_filename = OUTPUT_DIR / "19_pairplot_log_features.png"
        pp.fig.savefig(pp_filename, bbox_inches="tight", dpi=150)
        plt.close(pp.fig)

    # 20. Stacked bar: Loan Status proportion by Property Area
    if "Property_Area" in df.columns:
        prop = pd.crosstab(df["Property_Area"], df["Loan_Status_Label"], normalize="index")
        fig = prop.plot(kind="bar", stacked=True, figsize=(8, 6)).get_figure()
        plt.title("Proportion of Loan Status by Property Area")
        save_fig(fig, "20_stacked_prop_loanstatus_by_propertyarea")

    print("Visualizations saved.")






# ---------------------------------- Modelling ---------------------------------------

def prepare_model_data(df: pd.DataFrame):
    df = df.copy()

    # Drop identifiers
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Ensure target exists
    if "Loan_Status" not in df.columns:
        raise ValueError("Loan_Status target column not found in dataframe.")

    # Separate X and y
    y = df["Loan_Status"].astype(int)
    X = df.drop(columns=["Loan_Status", "Loan_Status_Label"], errors=True)

    # Categorical columns to encode
    cat_cols = X.select_dtypes(include=[object]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # One-hot encode categoricals (drop_first to avoid multicollinearity)
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Fill any remaining NA
    X_encoded = X_encoded.fillna(X_encoded.median())

    return X_encoded, y


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        cr = classification_report(y_test, preds)

        results[name] = {"model": model, "accuracy": acc, "confusion_matrix": cm, "report": cr}

        print(f"{name} Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", cr)

        # Save confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {name}")
        save_fig(fig, f"confusion_matrix_{name.lower()}")

    return results






# -------------------------------- Main ----------------------------------

def main():
    print("Starting Credit Risk Prediction pipeline...")

    # Load
    df = load_data(DATA_PATH)

    # Quick overview
    overview(df)

    # Clean missing values
    df_clean = handle_missing_values(df)

    # Feature engineering
    df_feat = feature_engineering(df_clean)

    # Visualizations
    create_visualizations(df_feat)

    # Modelling data
    X, y = prepare_model_data(df_feat)

    # Train & evaluate
    results = train_and_evaluate(X, y)

    # Summarize best model
    best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_acc = results[best_model_name]["accuracy"]
    print(f"\nBest model: {best_model_name} with accuracy {best_acc:.4f}")

    print("\nAll done. Check the 'output' folder next to your dataset for saved visualizations and confusion matrices.")


if __name__ == "__main__":
    main()










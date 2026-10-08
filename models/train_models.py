import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "hostel_complaints.csv"
OUT = ROOT / "models"
df = pd.read_csv(DATA)

features = ["Hostel_Block","Complaint_Type","Location","Severity","Season",
            "Days_Pending","Previous_Complaints","Affected_Rooms","Staff_Available"]
cat = ["Hostel_Block","Complaint_Type","Location","Severity","Season"]
num = ["Days_Pending","Previous_Complaints","Affected_Rooms","Staff_Available"]

prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")

X = df[features]
y = df["Priority"]
Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)

j48_like = Pipeline([("prep", prep), ("model", DecisionTreeClassifier(max_depth=6, random_state=42))])
j48_like.fit(Xtr,ytr)
joblib.dump(j48_like, OUT/"j48_like.joblib")

# GaussianNB needs dense encoded matrix
prep_dense = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)], remainder="passthrough")
nb = Pipeline([("prep", prep_dense), ("model", GaussianNB())])
nb.fit(Xtr,ytr)
joblib.dump(nb, OUT/"naive_bayes.joblib")

reg = Pipeline([("prep", prep_dense), ("model", LinearRegression())])
reg.fit(Xtr, df.loc[Xtr.index, "Resolution_Time_Days"])
joblib.dump(reg, OUT/"linear_regression.joblib")

# KMeans on one-hot + numeric features
Z = prep_dense.fit_transform(X)
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(Z)
joblib.dump((prep_dense, km), OUT/"kmeans.joblib")

print("Models trained successfully.")

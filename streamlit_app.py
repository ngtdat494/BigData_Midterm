import streamlit as st
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import quote_plus
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="🏠 Phân tích Giá Nhà - MongoDB + Streamlit", layout="wide")
st.title("🏠 Ứng dụng Phân tích Dữ liệu Nhà ở")
st.markdown("Bài tập giữa kỳ • Dữ liệu từ MongoDB Atlas")

# ===== 1. Load dữ liệu từ MongoDB Atlas =====
@st.cache_data
def load_housing_data():
    username = quote_plus("User")
    password = quote_plus("123456@Zz")
    uri = f"mongodb+srv://{username}:{password}@cluster0.8ugfq6t.mongodb.net/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(uri)
    db = client["housing"]
    col = db["housing_data"]

    data = list(col.find({}, {"_id": 0}))
    df = pd.DataFrame(data)

    bool_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    for colname in bool_cols:
        df[colname] = df[colname].map({"yes": 1, "no": 0})

    df["furnishingstatus"] = df["furnishingstatus"].map({"unfurnished": 0, "semi-furnished": 1, "furnished": 2})
    return df

# ===== 2. Tải dữ liệu và sidebar lọc =====
st.sidebar.header("🎛️ Bộ lọc dữ liệu")
df = load_housing_data()

unique_stories = sorted(df["stories"].unique())
story_filter = st.sidebar.multiselect("Chọn số tầng:", unique_stories, default=unique_stories)

furnishing_map = {0: "Unfurnished", 1: "Semi-furnished", 2: "Furnished"}
df["furnishingstatus_label"] = df["furnishingstatus"].map(furnishing_map)
unique_furnish = df["furnishingstatus_label"].unique()
furnish_filter = st.sidebar.multiselect("Tình trạng nội thất:", unique_furnish, default=unique_furnish)

df_filtered = df[df["stories"].isin(story_filter)]
df_filtered = df_filtered[df_filtered["furnishingstatus_label"].isin(furnish_filter)]

# ===== 3. Tổng quan dữ liệu =====
st.subheader("📌 Tổng quan dữ liệu sau lọc")
col1, col2, col3 = st.columns(3)
col1.metric("Số mẫu", f"{len(df_filtered):,}")
col2.metric("Giá TB", f"{df_filtered['price'].mean():,.0f}")
col3.metric("Diện tích TB", f"{df_filtered['area'].mean():,.0f} sqft")

st.markdown("---")

# ===== 3.1 Dự đoán Giá Nhà bằng Hồi Quy Tuyến Tính =====
st.subheader("🤖 3.1 Dự đoán Giá Nhà (Linear Regression)")

# Chọn các cột dùng làm đặc trưng
feature_cols = ["area", "bedrooms", "bathrooms", "stories", "parking", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
X = df_filtered[feature_cols]
y = df_filtered["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.write(f"**🎯 R² Score:** {r2:.2f}")
st.write(f"**📉 Mean Squared Error:** {mse:,.0f}")

# Hiển thị biểu đồ thực tế vs dự đoán
fig_pred, ax_pred = plt.subplots()
ax_pred.scatter(y_test, y_pred, alpha=0.6)
ax_pred.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax_pred.set_xlabel("Giá thực tế")
ax_pred.set_ylabel("Giá dự đoán")
ax_pred.set_title("Giá Nhà: Thực tế vs Dự đoán")
st.pyplot(fig_pred)

st.markdown("---")

# ===== 4. Bảng dữ liệu =====
st.subheader("📄 Dữ liệu nhà ở (Top 10)")
st.dataframe(df_filtered.head(10), use_container_width=True)

# ===== 5. Biểu đồ phân tích =====
st.subheader("📊 Biểu đồ phân tích")
c1, c2 = st.columns(2)

with c1:
    st.write("### Phân phối Giá Nhà")
    fig1, ax1 = plt.subplots()
    sns.histplot(df_filtered["price"], kde=True, ax=ax1, color="skyblue")
    ax1.set_xlabel("Giá nhà")
    st.pyplot(fig1)

with c2:
    st.write("### Phân phối Diện Tích")
    fig2, ax2 = plt.subplots()
    sns.histplot(df_filtered["area"], kde=True, ax=ax2, color="salmon")
    ax2.set_xlabel("Diện tích (sqft)")
    st.pyplot(fig2)

st.write("### 🔥 Ma trận tương quan")
fig3, ax3 = plt.subplots(figsize=(10, 6))
corr = df_filtered[["price", "area", "bedrooms", "bathrooms", "stories", "parking"] + bool_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
st.pyplot(fig3)

st.write("### 🛏️ Boxplot: Giá nhà theo số phòng ngủ")
fig4, ax4 = plt.subplots()
sns.boxplot(data=df_filtered, x="bedrooms", y="price", palette="Set2", ax=ax4)
ax4.set_xlabel("Số phòng ngủ")
ax4.set_ylabel("Giá nhà")
st.pyplot(fig4)

st.write("### 📈 Scatter: Diện tích vs Giá nhà")
fig5, ax5 = plt.subplots()
sns.scatterplot(data=df_filtered, x="area", y="price", hue="furnishingstatus_label", alpha=0.6, ax=ax5)
ax5.set_xlabel("Diện tích (sqft)")
ax5.set_ylabel("Giá nhà")
ax5.legend(title="Nội thất")
st.pyplot(fig5)

st.markdown("---")
st.success("✅ Ứng dụng đã hoàn thiện với phần dự đoán giá nhà!")

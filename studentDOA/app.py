"""Dashboard eksplorasi dan prediksi status mahasiswa.

Jalankan dari folder ini: streamlit run app.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
TARGET = "Target"
PREPARED_FILE = BASE_DIR / "student_ml_dataset.csv"
SOURCE_FILE = BASE_DIR / "student_doa.csv"
MODEL_FILE = BASE_DIR / "student_outcome_pipeline.pkl"

st.set_page_config(page_title="Student Outcome Analysis", page_icon="🎓", layout="wide")
st.title("Student Data: Eksplorasi dan Prediksi Status")
st.caption(
    "Pahami data mahasiswa, tinjau ringkasan akademik, dan gunakan model untuk memperkirakan "
    "status Graduate, Dropout, atau Enrolled."
)


@st.cache_data
def load_data():
    path = PREPARED_FILE if PREPARED_FILE.exists() else SOURCE_FILE
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_outputs():
    metrics_path = BASE_DIR / "model_metrics.json"
    importance_path = BASE_DIR / "feature_importance.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
    return metrics, importance


try:
    data = load_data()
    if TARGET not in data.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan.")
except Exception as exc:
    st.error(f"Dataset gagal dibaca: {exc}")
    st.stop()

model = load_model()
metrics, feature_importance = load_outputs()
labels = data[TARGET].astype(str)
class_counts = labels.value_counts()

metric_cols = st.columns(4)
metric_cols[0].metric("Jumlah mahasiswa", f"{len(data):,}")
metric_cols[1].metric("Jumlah fitur", f"{data.shape[1] - 1}")
metric_cols[2].metric("Status terbanyak", str(class_counts.index[0]))
metric_cols[3].metric(
    "Akurasi data uji", f"{metrics['accuracy']:.1%}" if "accuracy" in metrics else "Belum dilatih"
)

tabs = st.tabs(["1 · Data", "2 · Ringkasan akademik", "3 · Machine Learning"])

with tabs[0]:
    st.subheader("Dataset mahasiswa")
    st.write(f"Ukuran: **{data.shape[0]:,} baris × {data.shape[1]} kolom**")
    st.markdown(
        "Data mencakup atribut pendaftaran dan demografi, kondisi ekonomi, hasil mata kuliah "
        "semester 1 dan 2, indikator ekonomi, serta target `Target`. Banyak kolom merupakan "
        "kode angka; gunakan data dictionary untuk mengetahui arti setiap kode."
    )
    a, b = st.columns(2)
    a.metric("Nilai kosong", f"{int(data.isna().sum().sum()):,}")
    b.metric("Baris duplikat", f"{int(data.duplicated().sum()):,}")
    with st.expander("Kolom dan tipe data"):
        st.dataframe(
            pd.DataFrame({"Kolom": data.columns, "Tipe": data.dtypes.astype(str).values}),
            hide_index=True,
            use_container_width=True,
        )
    st.dataframe(data.head(100), hide_index=True, use_container_width=True)
    st.download_button(
        "Unduh dataset CSV",
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name="student_ml_dataset.csv",
        mime="text/csv",
    )

with tabs[1]:
    st.subheader("Gambaran status dan hasil akademik")
    left, right = st.columns(2)
    with left:
        st.markdown("**Jumlah mahasiswa per status**")
        st.bar_chart(class_counts.rename_axis("Status").to_frame("Jumlah"))
    with right:
        st.markdown("**Rata-rata nilai per semester berdasarkan status**")
        grade_summary = data.groupby(TARGET)[
            ["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]
        ].mean()
        grade_summary.columns = ["Semester 1", "Semester 2"]
        st.bar_chart(grade_summary)

    st.markdown("**Rata-rata mata kuliah disetujui per semester dan status**")
    approved_summary = data.groupby(TARGET)[
        ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"]
    ].mean()
    approved_summary.columns = ["Semester 1", "Semester 2"]
    st.bar_chart(approved_summary)

    st.markdown("**Distribusi usia saat mendaftar**")
    age_data = data[["Age at enrollment", TARGET]].rename(
        columns={"Age at enrollment": "Usia saat mendaftar", TARGET: "Status"}
    )
    st.vega_lite_chart(
        age_data,
        {
            "mark": {"type": "bar", "opacity": 0.7},
            "encoding": {
                "x": {"field": "Usia saat mendaftar", "type": "quantitative", "bin": {"maxbins": 25}},
                "y": {"aggregate": "count", "type": "quantitative", "title": "Jumlah mahasiswa"},
                "color": {"field": "Status", "type": "nominal"},
            },
        },
        use_container_width=True,
    )

with tabs[2]:
    st.subheader("Prediksi status mahasiswa")
    if model is None:
        st.warning(
            "Model belum tersedia. Jalankan `python data_prep.py`, lalu `python ml_model.py` "
            "dari folder `studentDOA` untuk membuat model dan hasil evaluasi."
        )
    else:
        st.info(
            "Model Random Forest dilatih pada data latih dan dievaluasi pada data uji. "
            "Prediksi adalah perkiraan pembelajaran, bukan kepastian atau keputusan administratif."
        )
        student_index = st.selectbox(
            "Pilih baris mahasiswa",
            options=data.index.tolist(),
            format_func=lambda idx: f"Baris {idx + 1} · status aktual: {labels.iloc[idx]}",
        )
        student = data.iloc[[student_index]]
        features = student.drop(columns=[TARGET])
        predicted_label = str(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        probability_frame = pd.DataFrame(
            {"Status": model.classes_, "Probabilitas": probabilities}
        ).sort_values("Probabilitas", ascending=False)
        m1, m2 = st.columns(2)
        m1.metric("Prediksi model", predicted_label)
        m2.metric("Status aktual", str(student[TARGET].iloc[0]))
        st.markdown("**Probabilitas setiap status**")
        st.bar_chart(probability_frame.set_index("Status"))
        with st.expander("Lihat data mahasiswa"):
            st.dataframe(student, hide_index=True, use_container_width=True)

        if metrics:
            st.markdown("**Evaluasi pada data uji**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            m2.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}")
            m3.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
            report = pd.DataFrame(metrics["classification_report"]).transpose().rename_axis("Kelas")
            st.dataframe(report.round(3), use_container_width=True)
            st.markdown("**Confusion matrix (baris aktual, kolom prediksi)**")
            st.dataframe(
                pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=metrics["classes"],
                    columns=metrics["classes"],
                ).rename_axis("Aktual"),
                use_container_width=True,
            )
        if not feature_importance.empty:
            st.markdown("**15 fitur dengan importance tertinggi**")
            importance = feature_importance.head(15).set_index("feature")
            st.bar_chart(importance["importance"])

st.divider()
st.caption(
    "Kolom kode kategori diproses dengan One-Hot Encoding di pipeline model. "
    "Imputasi dan encoding dipelajari dari data latih untuk menghindari kebocoran data."
)

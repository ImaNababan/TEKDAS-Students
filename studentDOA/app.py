"""Dashboard eksplorasi, ringkasan, prediksi, dan evaluasi status mahasiswa.

Jalankan dari folder ini dengan: streamlit run app.py
"""
from pathlib import Path
import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
TARGET = "Target"
PREPARED_FILE = BASE_DIR / "student_ml_dataset.csv"
SOURCE_FILE = BASE_DIR / "student_doa.csv"
TRAIN_FILE = BASE_DIR / "student_train.csv"
TEST_FILE = BASE_DIR / "student_test.csv"
MODEL_FILE = BASE_DIR / "student_outcome_pipeline.pkl"
STATUS_COLORS = {"Graduate": "#2E8B57", "Dropout": "#D64545", "Enrolled": "#2878B5"}
STATUS_ORDER = ["Graduate", "Dropout", "Enrolled"]
CAT_COLUMNS = [
    "Marital status", "Application mode", "Course", "Daytime/evening attendance",
    "Previous qualification", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "International",
]
BOOLEAN_LABELS = {
    "Gender": {"0": "Perempuan", "1": "Laki-laki"},
    "Daytime/evening attendance": {"0": "Sore", "1": "Siang"},
    "Displaced": {"0": "Tidak", "1": "Ya"},
    "Educational special needs": {"0": "Tidak", "1": "Ya"},
    "Debtor": {"0": "Tidak", "1": "Ya"},
    "Tuition fees up to date": {"0": "Belum lunas", "1": "Lunas"},
    "Scholarship holder": {"0": "Tidak", "1": "Ya"},
    "International": {"0": "Tidak", "1": "Ya"},
}
FRIENDLY = {
    "Marital status": {"1": "Lajang", "2": "Menikah", "3": "Duda/janda", "4": "Bercerai", "5": "Fakta de facto", "6": "Pisah secara hukum"},
    "Application mode": {"1": "Fase 1 – umum", "2": "Fase 2 – umum", "5": "Fase 1 – khusus", "7": "Pemegang ijazah luar negeri", "10": "Fase 1 – umum (kontingen)", "15": "Fase 1 – teknis", "16": "Fase 1 – teknis (kontingen)", "17": "Fase 2 – teknis", "18": "Fase 3 – teknis", "26": "Fase 1 – khusus (kontingen)", "27": "Fase 2 – khusus", "39": "Fase 1 – lulusan sebelumnya", "42": "Fase 1 – lulusan sebelumnya (kontingen)", "43": "Fase 2 – lulusan sebelumnya", "44": "Fase 3 – lulusan sebelumnya", "51": "Fase 1 – perubahan program", "53": "Fase 1 – perubahan program (kontingen)", "57": "Fase 2 – perubahan program", "招生": "Pendaftaran lainnya"},
    "Course": {"33": "Keperawatan", "171": "Manajemen (siang)", "8014": "Layanan Sosial (siang)", "9003": "Agronomi", "9070": "Desain Komunikasi", "9085": "Keperawatan (vokasi)", "9119": "Manajemen (malam)", "9130": "Teknologi Informatika", "9147": "Desain Produk", "9238": "Pendidikan Dasar", "9254": "Manajemen Pemasaran", "9500": "Jurnalisme", "9556": "Pendidikan Sosial", "9670": "Layanan Sosial (malam)", "9773": "Pariwisata", "9853": "Akuntansi", "9991": "Manajemen (vokasi)"},
    "Displaced": {},
}
FRIENDLY["Nacionality"] = {"1": "Portugis", "2": "Jerman", "6": "Spanyol", "11": "Italia", "14": "Belanda", "17": "Inggris", "21": "Brasil", "22": "Angola", "24": "Tanjung Verde", "25": "Guinea", "26": "Mozambik", "32": "Turki", "41": "Rusia", "62": "Rumania", "100": "Meksiko", "101": "Ukraina", "103": "Kolombia", "105": "Kuba", "108": "Sao Tome dan Principe", "109": "Timor-Leste"}


def model_input(frame):
    """Match kategori dengan tipe string yang dipakai ketika pipeline dilatih."""
    result = frame.copy()
    for column in CAT_COLUMNS:
        if column in result:
            result[column] = result[column].astype("string")
    return result

st.set_page_config(page_title="Student Outcome Analysis", page_icon="🎓", layout="wide")


@st.cache_data
def load_data():
    if PREPARED_FILE.exists():
        frame = pd.read_csv(PREPARED_FILE)
    else:
        frame = pd.read_csv(SOURCE_FILE, sep=";", encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    return frame


@st.cache_data
def load_split(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None


@st.cache_data
def load_outputs():
    mp, ip = BASE_DIR / "model_metrics.json", BASE_DIR / "feature_importance.csv"
    metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    importance = pd.read_csv(ip) if ip.exists() else pd.DataFrame()
    return metrics, importance


def display_label(column, value):
    key = str(value)
    return BOOLEAN_LABELS.get(column, {}).get(key, FRIENDLY.get(column, {}).get(key, f"Kategori {key}"))


def present(frame):
    """Buat salinan berlabel untuk tampilan; data model tetap berupa kode asli."""
    shown = frame.copy()
    for column in CAT_COLUMNS:
        if column in shown:
            shown[column] = shown[column].map(lambda value: display_label(column, value) if pd.notna(value) else value)
    return shown


def status_color():
    return alt.Scale(domain=STATUS_ORDER, range=[STATUS_COLORS[s] for s in STATUS_ORDER])


def chart_note(text):
    st.caption(text)


def render_status_bar(frame, category, title, y_title="Jumlah mahasiswa", normalize=False):
    grouped = frame.groupby([category, TARGET], observed=False).size().rename("Jumlah").reset_index()
    if normalize:
        grouped["Jumlah"] = grouped["Jumlah"] / grouped.groupby(category)["Jumlah"].transform("sum")
    chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X(f"{category}:N", title=category, sort="-y"),
        y=alt.Y("Jumlah:Q", title="Proporsi" if normalize else y_title, axis=alt.Axis(format="%") if normalize else alt.Undefined),
        color=alt.Color(f"{TARGET}:N", scale=status_color(), sort=STATUS_ORDER, title="Status"),
        tooltip=[category, TARGET, alt.Tooltip("Jumlah:Q", format=".1%" if normalize else ",.0f")],
    )
    st.altair_chart(chart.properties(title=title), use_container_width=True)


try:
    data = load_data()
    if TARGET not in data:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan.")
except Exception as exc:
    st.error(f"Dataset gagal dibaca: {exc}")
    st.stop()

model = load_model()
train_data = load_split(TRAIN_FILE)
test_data = load_split(TEST_FILE)
metrics, feature_importance = load_outputs()

st.title("Student Data: Eksplorasi dan Prediksi Status")
st.caption("Pahami data mahasiswa, tinjau ringkasan akademik, dan gunakan model untuk memperkirakan status.")

# Ringkasan utama tetap terlihat di semua halaman seperti bagian pembuka dashboard.
class_counts = data[TARGET].astype(str).value_counts()
overview = st.columns(4)
overview[0].metric("Jumlah mahasiswa", f"{len(data):,}")
overview[1].metric("Jumlah fitur", f"{data.shape[1] - 1}")
overview[2].metric("Status terbanyak", class_counts.index[0] if not class_counts.empty else "-" )
overview[3].metric("Akurasi data uji", f"{metrics['accuracy']:.1%}" if "accuracy" in metrics else "Belum dilatih")

page = st.sidebar.radio("Navigasi", ["Data", "Ringkasan Akademik", "Prediksi", "Evaluasi Model"])
st.sidebar.divider()
st.sidebar.subheader("Filter data")
filtered = data.copy()
filter_specs = [
    ("Gender", "Gender"), ("Scholarship holder", "Beasiswa"),
    ("Tuition fees up to date", "Status SPP"),
]
for col, title in filter_specs:
    if col in filtered:
        values = sorted(filtered[col].dropna().astype(str).unique(), key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else v))
        choices = st.sidebar.multiselect(title, values, default=values, format_func=lambda v, c=col: display_label(c, v))
        filtered = filtered[filtered[col].astype(str).isin(choices)]
if "Age at enrollment" in filtered:
    ages = pd.to_numeric(filtered["Age at enrollment"], errors="coerce").dropna()
    if not ages.empty:
        age_range = st.sidebar.slider("Rentang usia saat mendaftar", int(ages.min()), int(ages.max()), (int(ages.min()), int(ages.max())))
        filtered = filtered[pd.to_numeric(filtered["Age at enrollment"], errors="coerce").between(*age_range)]

if page == "Data":
    st.subheader("Dataset mahasiswa")
    st.write(f"Ukuran setelah filter: **{len(filtered):,} baris × {filtered.shape[1]} kolom**")
    a, b = st.columns(2)
    a.metric("Nilai kosong", f"{int(filtered.isna().sum().sum()):,}")
    b.metric("Baris duplikat", f"{int(filtered.duplicated().sum()):,}")
    st.markdown("**Jumlah mahasiswa menurut status**")
    count = filtered[TARGET].value_counts().rename_axis(TARGET).reset_index(name="Jumlah")
    st.altair_chart(alt.Chart(count).mark_bar().encode(
        x=alt.X(f"{TARGET}:N", sort=STATUS_ORDER, title="Status"), y=alt.Y("Jumlah:Q"),
        color=alt.Color(f"{TARGET}:N", scale=status_color(), sort=STATUS_ORDER, legend=None),
        tooltip=[TARGET, "Jumlah"]), use_container_width=True)
    chart_note("Grafik menunjukkan jumlah mahasiswa pada setiap status setelah filter sidebar diterapkan.")
    with st.expander("Kolom dan tipe data"):
        st.dataframe(pd.DataFrame({"Kolom": data.columns, "Tipe": data.dtypes.astype(str).values}), hide_index=True, use_container_width=True)
    st.dataframe(present(filtered.head(500)), hide_index=True, use_container_width=True)
    st.download_button("Unduh dataset terfilter CSV", data=filtered.to_csv(index=False).encode("utf-8-sig"), file_name="student_ml_dataset_filtered.csv", mime="text/csv")

elif page == "Ringkasan Akademik":
    st.subheader("Gambaran status dan hasil akademik")
    left, right = st.columns(2)
    with left:
        st.markdown("**Jumlah mahasiswa per status**")
        count = filtered[TARGET].value_counts().rename_axis(TARGET).reset_index(name="Jumlah")
        st.altair_chart(alt.Chart(count).mark_bar().encode(x=alt.X(f"{TARGET}:N", sort=STATUS_ORDER), y="Jumlah:Q", color=alt.Color(f"{TARGET}:N", scale=status_color(), sort=STATUS_ORDER, legend=None), tooltip=[TARGET, "Jumlah"]), use_container_width=True)
        chart_note("Jumlah memperlihatkan komposisi status pada kelompok yang lolos seluruh filter.")
    with right:
        st.markdown("**Tingkat dropout menurut status SPP dan beasiswa**")
        breakdown = []
        for col, name in [("Tuition fees up to date", "Status SPP"), ("Scholarship holder", "Beasiswa")]:
            if col in filtered:
                temp = filtered.groupby(col, dropna=False)[TARGET].apply(lambda s: (s == "Dropout").mean()).reset_index(name="Tingkat dropout")
                temp["Faktor"] = name
                temp["Kategori"] = temp[col].map(lambda x: display_label(col, x))
                breakdown.append(temp[["Faktor", "Kategori", "Tingkat dropout"]])
        if breakdown:
            rates = pd.concat(breakdown, ignore_index=True)
            st.altair_chart(alt.Chart(rates).mark_bar().encode(x=alt.X("Kategori:N", title="Kategori"), y=alt.Y("Tingkat dropout:Q", axis=alt.Axis(format="%")), color=alt.Color("Faktor:N", title="Faktor"), column="Faktor:N", tooltip=["Faktor", "Kategori", alt.Tooltip("Tingkat dropout:Q", format=".1%")]), use_container_width=True)
            chart_note("Persentase dropout dihitung dari mahasiswa di setiap kategori SPP atau beasiswa.")
    if {"Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"}.issubset(filtered):
        grades = filtered.groupby(TARGET)[["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]].mean().reset_index().melt(TARGET, var_name="Semester", value_name="Rata-rata nilai")
        st.markdown("**Rata-rata nilai semester menurut status**")
        st.altair_chart(alt.Chart(grades).mark_bar().encode(x=alt.X("Semester:N"), y=alt.Y("Rata-rata nilai:Q"), color=alt.Color(f"{TARGET}:N", scale=status_color(), sort=STATUS_ORDER), xOffset=f"{TARGET}:N", tooltip=["Semester", TARGET, alt.Tooltip("Rata-rata nilai:Q", format=".2f")]), use_container_width=True)
        chart_note("Grafik membandingkan rata-rata nilai semester pertama dan kedua untuk tiap status.")
    if {"Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"}.issubset(filtered):
        approved = filtered.groupby(TARGET)[["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"]].mean().reset_index().melt(TARGET, var_name="Semester", value_name="Rata-rata mata kuliah disetujui")
        st.markdown("**Rata-rata mata kuliah disetujui**")
        st.altair_chart(alt.Chart(approved).mark_bar().encode(x=alt.X("Semester:N"), y="Rata-rata mata kuliah disetujui:Q", color=alt.Color(f"{TARGET}:N", scale=status_color(), sort=STATUS_ORDER), xOffset=f"{TARGET}:N", tooltip=["Semester", TARGET, alt.Tooltip("Rata-rata mata kuliah disetujui:Q", format=".2f")]), use_container_width=True)
        chart_note("Rata-rata mata kuliah yang lulus memberi gambaran kemajuan akademik per semester dan status.")
    if "Age at enrollment" in filtered:
        ages = filtered[["Age at enrollment", TARGET]].rename(columns={"Age at enrollment": "Usia"})
        st.markdown("**Distribusi usia saat mendaftar**")
        st.altair_chart(alt.Chart(ages).mark_bar(opacity=.8).encode(x=alt.X("Usia:Q", bin=alt.Bin(maxbins=25)), y=alt.Y("count():Q", title="Jumlah mahasiswa"), color=alt.Color(f"{TARGET}:N", scale=status_color(), sort=STATUS_ORDER), tooltip=[alt.Tooltip("Usia:Q", bin=alt.Bin(maxbins=25)), TARGET, alt.Tooltip("count():Q", title="Jumlah")]), use_container_width=True)
        chart_note("Distribusi ini memperlihatkan rentang usia pendaftar dan statusnya untuk data terfilter.")
    # Concise data-derived narrative, with safe handling for empty filters.
    if filtered.empty:
        st.info("Tidak ada data yang sesuai dengan filter, sehingga insight belum dapat dihitung.")
    else:
        st.markdown("**Insight otomatis**")
        total_dropout = (filtered[TARGET] == "Dropout").mean()
        st.write(f"1. Tingkat dropout pada kelompok terfilter adalah **{total_dropout:.1%}**.")
        insights = []
        for col, name in [("Tuition fees up to date", "status SPP"), ("Scholarship holder", "beasiswa")]:
            if col in filtered:
                rates = filtered.groupby(col)[TARGET].apply(lambda s: (s == "Dropout").mean()).dropna()
                if not rates.empty:
                    high, low = rates.idxmax(), rates.idxmin()
                    insights.append(f"{name}: dropout tertinggi pada **{display_label(col, high)}** ({rates.max():.1%}) dan terendah pada **{display_label(col, low)}** ({rates.min():.1%}).")
        for i, line in enumerate(insights[:2], start=2):
            st.write(f"{i}. {line}")
        if "Curricular units 2nd sem (approved)" in filtered:
            approved_means = filtered.groupby(TARGET)["Curricular units 2nd sem (approved)"].mean().dropna()
            if not approved_means.empty:
                st.write(f"4. Rata-rata mata kuliah semester 2 yang disetujui paling tinggi pada **{approved_means.idxmax()}** ({approved_means.max():.2f}).")

elif page == "Prediksi":
    st.subheader("3 · Machine Learning — Prediksi status mahasiswa")
    if model is None:
        st.warning("Model belum tersedia. Jalankan `python data_prep.py`, lalu `python ml_model.py` dari folder `studentDOA`.")
    else:
        st.info("Prediksi adalah perkiraan model untuk pembelajaran, bukan kepastian atau keputusan administratif.")
        mode = st.radio("Mode prediksi", ["Pilih baris data uji", "Simulasi what-if"], horizontal=True)
        feature_columns = [column for column in data.columns if column != TARGET]
        if mode == "Pilih baris data uji":
            if test_data.empty:
                st.error("student_test.csv tidak ditemukan. Jalankan data_prep.py terlebih dahulu.")
                st.stop()
            idx = st.selectbox("Pilih baris dari student_test.csv", test_data.index.tolist(), format_func=lambda i: f"Baris uji {i + 1} · aktual: {test_data.iloc[i][TARGET]}")
            row = test_data.iloc[[idx]]
            features = row.drop(columns=[TARGET])
        else:
            source_train = train_data if not train_data.empty else data
            features = pd.DataFrame([{col: (source_train[col].median() if col not in CAT_COLUMNS else source_train[col].mode(dropna=True).iloc[0]) for col in feature_columns}])
            with st.form("what_if_form"):
                st.caption("Fitur yang tidak tampil di bawah diisi dengan median (numerik) atau modus (kategori) data latih.")
                cols = st.columns(2)
                controllable = [
                    ("Curricular units 1st sem (grade)", "Nilai semester 1", "number"),
                    ("Curricular units 2nd sem (grade)", "Nilai semester 2", "number"),
                    ("Curricular units 1st sem (approved)", "Mata kuliah disetujui semester 1", "number"),
                    ("Curricular units 2nd sem (approved)", "Mata kuliah disetujui semester 2", "number"),
                    ("Age at enrollment", "Usia saat mendaftar", "number"),
                ]
                for i, (col, title, _) in enumerate(controllable):
                    if col in features:
                        numeric = pd.to_numeric(source_train[col], errors="coerce").dropna()
                        low, high = int(numeric.min()), int(numeric.max())
                        default = float(pd.to_numeric(features[col], errors="coerce").iloc[0])
                        step = 0.1 if "grade" in col else 1.0
                        if step == 1.0:
                            low, high, default, step = int(low), int(high), int(round(default)), 1
                        val = cols[i % 2].slider(title, float(low) if step < 1 else low, float(high) if step < 1 else high, min(max(default, low), high), step=step, key=f"what_{col}")
                        features.loc[0, col] = val
                for col, title in [("Tuition fees up to date", "Status SPP"), ("Scholarship holder", "Beasiswa")]:
                    if col in features:
                        opts = sorted(source_train[col].dropna().astype(str).unique())
                        choice = st.selectbox(title, opts, format_func=lambda v, c=col: display_label(c, v), key=f"what_{col}")
                        features.loc[0, col] = choice
                submitted = st.form_submit_button("Hitung prediksi", type="primary")
            if not submitted:
                st.stop()
            row = features.copy()
        features = model_input(features)
        predicted = str(model.predict(features)[0])
        probs = model.predict_proba(features)[0]
        st.metric("Prediksi model", predicted)
        prob_frame = pd.DataFrame({"Status": model.classes_, "Probabilitas": probs})
        st.markdown("**Probabilitas tiap status**")
        st.altair_chart(alt.Chart(prob_frame).mark_bar().encode(x=alt.X("Probabilitas:Q", axis=alt.Axis(format="%")), y=alt.Y("Status:N", sort="-x"), color=alt.Color("Status:N", scale=status_color(), sort=STATUS_ORDER, legend=None), tooltip=["Status", alt.Tooltip("Probabilitas:Q", format=".1%")]), use_container_width=True)
        chart_note("Panjang batang adalah probabilitas model untuk masing-masing status pada input yang dipilih.")
        with st.expander("Lihat fitur yang digunakan"):
            st.dataframe(present(row), hide_index=True, use_container_width=True)

elif page == "Evaluasi Model":
    st.subheader("Evaluasi model pada data uji")
    if not metrics:
        st.warning("Hasil evaluasi belum tersedia. Jalankan `python ml_model.py` dari folder `studentDOA`.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c2.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}")
        c3.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        report = pd.DataFrame(metrics["classification_report"]).transpose().rename_axis("Kelas")
        st.dataframe(report.round(3), use_container_width=True)
        st.caption("Metrik dihitung dari prediksi pada bagian data uji yang tidak digunakan saat pelatihan.")
        st.markdown("**Confusion matrix (baris aktual, kolom prediksi)**")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=metrics["classes"], columns=metrics["classes"]).rename_axis("Aktual"), use_container_width=True)
        st.caption("Matriks menunjukkan jumlah prediksi untuk setiap pasangan status aktual dan status model.")
    if not feature_importance.empty:
        st.markdown("**15 fitur dengan importance tertinggi**")
        importance = feature_importance.head(15).sort_values("importance")
        st.altair_chart(alt.Chart(importance).mark_bar().encode(x=alt.X("importance:Q", title="Importance"), y=alt.Y("feature:N", sort="-x", title="Fitur"), tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")]), use_container_width=True)
        chart_note("Importance menunjukkan kontribusi relatif fitur dalam model Random Forest, bukan hubungan sebab-akibat.")

st.sidebar.caption("Warna status: Graduate hijau · Dropout merah · Enrolled biru.")
st.divider()
st.caption("Kategori diberi label untuk tampilan saja; nilai kode asli tetap digunakan oleh pipeline prediksi.")
